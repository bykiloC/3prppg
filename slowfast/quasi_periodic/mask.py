#!/usr/bin/env python3

"""Strict 8x8 background assignment for the quasi-periodic branch."""

from collections import Counter
from pathlib import Path

import numpy as np


CURRENT_LABELS = {
    0: "background",
    1: "skin",
    2: "nose",
    3: "right_eye",
    4: "left_eye",
    5: "right_brow",
    6: "left_brow",
    7: "right_ear",
    8: "left_ear",
    9: "mouth_interior",
    10: "top_lip",
    11: "bottom_lip",
    12: "neck",
    13: "hair",
    14: "beard",
    15: "clothing",
    16: "glasses",
    17: "headwear",
    18: "facewear",
}

REGION_CLASSES = (
    ("background", (0, 13, 15, 17, 18)),
    ("upper_face", (2, 3, 4, 5, 6, 16)),
    ("mouth_beard", (9, 10, 11, 14)),
    ("skin_neck", (1, 7, 8, 12)),
)
NOSE_LABELS = (2,)
BROW_EYE_LABELS = (3, 4, 5, 6, 16)
CLASS_NAMES = tuple(name for name, _label_ids in REGION_CLASSES)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(REGION_CLASSES)
GRID_SIZE = 8
BACKGROUND_THRESHOLD = 0.999999
BROW_EYE_THRESHOLD = 0.02
NOSE_THRESHOLD = 0.20
MOUTH_BEARD_THRESHOLD = 0.20


def mask_label_path(video_path):
    path = Path(video_path)
    return path.with_name("{}_mlabel.npy".format(path.stem))


def _build_label_to_class():
    label_to_class = np.full(max(CURRENT_LABELS) + 1, -1, dtype=np.int16)
    seen = {}
    for class_id, (class_name, label_ids) in enumerate(REGION_CLASSES):
        for label_id in label_ids:
            if label_id in seen:
                raise ValueError(
                    "label {} maps to both {} and {}".format(
                        label_id, seen[label_id], class_name
                    )
                )
            seen[label_id] = class_name
            label_to_class[int(label_id)] = int(class_id)
    missing = [label_id for label_id in CURRENT_LABELS if label_to_class[label_id] < 0]
    if missing:
        raise ValueError("region mapping misses labels: {}".format(missing))
    return label_to_class


LABEL_TO_CLASS = _build_label_to_class()


def coarse_clip(mask_clip):
    if mask_clip.ndim != 3:
        raise ValueError("expected T,H,W mask clip, got {}".format(mask_clip.shape))
    coarse = np.full(mask_clip.shape, -1, dtype=np.int16)
    known = mask_clip < len(LABEL_TO_CLASS)
    coarse[known] = LABEL_TO_CLASS[mask_clip[known]]
    unknown_values = np.unique(mask_clip[coarse < 0])
    unknown = Counter({int(v): int((mask_clip == v).sum()) for v in unknown_values})
    return coarse, unknown


def region_ratios_for_grid(coarse, grid_size):
    frames, height, width = coarse.shape
    grid_h, grid_w = grid_size
    ratios = np.empty((frames, grid_h, grid_w, NUM_CLASSES), dtype=np.float32)
    for gy in range(grid_h):
        y0 = (gy * height) // grid_h
        y1 = ((gy + 1) * height) // grid_h
        for gx in range(grid_w):
            x0 = (gx * width) // grid_w
            x1 = ((gx + 1) * width) // grid_w
            block = coarse[:, y0:y1, x0:x1].reshape(frames, -1)
            block_pixels = float(block.shape[1])
            for class_id in range(NUM_CLASSES):
                ratios[:, gy, gx, class_id] = (
                    (block == class_id).sum(axis=1) / block_pixels
                )
    return ratios


def label_ratios_for_grid(mask_clip, label_ids, grid_size):
    frames, height, width = mask_clip.shape
    grid_h, grid_w = grid_size
    ratios = np.empty((frames, grid_h, grid_w), dtype=np.float32)
    label_ids_array = np.asarray(label_ids, dtype=np.uint8)
    for gy in range(grid_h):
        y0 = (gy * height) // grid_h
        y1 = ((gy + 1) * height) // grid_h
        for gx in range(grid_w):
            x0 = (gx * width) // grid_w
            x1 = ((gx + 1) * width) // grid_w
            block = mask_clip[:, y0:y1, x0:x1].reshape(frames, -1)
            ratios[:, gy, gx] = np.isin(block, label_ids_array).sum(axis=1) / float(
                block.shape[1]
            )
    return ratios


def assign_regions(
    ratios,
    brow_eye_ratio,
    nose_ratio,
    background_threshold,
    brow_eye_threshold,
    nose_threshold,
    mouth_beard_threshold,
):
    background_ratio = ratios[..., CLASS_TO_ID["background"]]
    mouth_ratio = ratios[..., CLASS_TO_ID["mouth_beard"]]

    assigned = np.full(
        background_ratio.shape, CLASS_TO_ID["skin_neck"], dtype=np.uint8
    )
    background_hit = background_ratio > float(background_threshold)
    brow_eye_hit = brow_eye_ratio > float(brow_eye_threshold)
    nose_hit = nose_ratio > float(nose_threshold)
    upper_hit = (~background_hit) & (brow_eye_hit | nose_hit)
    mouth_hit = (
        (~background_hit)
        & (~upper_hit)
        & (mouth_ratio > float(mouth_beard_threshold))
    )
    assigned[background_hit] = CLASS_TO_ID["background"]
    assigned[upper_hit] = CLASS_TO_ID["upper_face"]
    assigned[mouth_hit] = CLASS_TO_ID["mouth_beard"]
    return assigned


class BackgroundMaskCache:
    def __init__(self):
        self._mask_arrays = {}
        self._assignment_cache = {}

    def load_mask_clip(self, video_path, frame_indices):
        mask_path = mask_label_path(video_path)
        frame_tuple = tuple(
            int(x) for x in np.asarray(frame_indices, dtype=np.int64).tolist()
        )
        if mask_path not in self._mask_arrays:
            if not mask_path.is_file():
                raise FileNotFoundError("mask label not found: {}".format(mask_path))
            mask_arr = np.load(mask_path, mmap_mode="r")
            if mask_arr.ndim != 3 or mask_arr.dtype != np.uint8:
                raise ValueError(
                    "expected uint8 T,H,W mask, got {} {}: {}".format(
                        mask_arr.dtype, mask_arr.shape, mask_path
                    )
                )
            self._mask_arrays[mask_path] = mask_arr

        mask_arr = self._mask_arrays[mask_path]
        frame_idx = np.asarray(frame_tuple, dtype=np.int64)
        if np.any(frame_idx < 0) or np.any(frame_idx >= mask_arr.shape[0]):
            raise IndexError(
                "frame index out of bounds for {}: min={} max={} frames={}".format(
                    mask_path, frame_idx.min(), frame_idx.max(), mask_arr.shape[0]
                )
            )

        return np.asarray(mask_arr[frame_idx], dtype=np.uint8)

    def assignment_from_mask_clip(self, mask_clip):
        coarse, unknown = coarse_clip(mask_clip)
        if unknown:
            raise ValueError(
                "unknown mask labels in mask clip: {}".format(dict(unknown))
            )

        grid = (GRID_SIZE, GRID_SIZE)
        ratios = region_ratios_for_grid(coarse, grid)
        brow_eye_ratio = label_ratios_for_grid(mask_clip, BROW_EYE_LABELS, grid)
        nose_ratio = label_ratios_for_grid(mask_clip, NOSE_LABELS, grid)
        assigned = assign_regions(
            ratios,
            brow_eye_ratio=brow_eye_ratio,
            nose_ratio=nose_ratio,
            background_threshold=BACKGROUND_THRESHOLD,
            brow_eye_threshold=BROW_EYE_THRESHOLD,
            nose_threshold=NOSE_THRESHOLD,
            mouth_beard_threshold=MOUTH_BEARD_THRESHOLD,
        )
        return assigned

    def load_assignment(self, video_path, frame_indices):
        mask_path = mask_label_path(video_path)
        frame_tuple = tuple(
            int(x) for x in np.asarray(frame_indices, dtype=np.int64).tolist()
        )
        key = (mask_path, frame_tuple)
        if key in self._assignment_cache:
            return self._assignment_cache[key]

        mask_clip = self.load_mask_clip(video_path, frame_tuple)
        assigned = self.assignment_from_mask_clip(mask_clip)
        self._assignment_cache[key] = assigned
        return assigned

    def background_mask(self, video_path, frame_indices):
        assigned = self.load_assignment(video_path, frame_indices)
        return (assigned == CLASS_TO_ID["background"]).astype(np.float32)

    def background_mask_from_clip(self, mask_clip):
        assigned = self.assignment_from_mask_clip(np.asarray(mask_clip, dtype=np.uint8))
        return (assigned == CLASS_TO_ID["background"]).astype(np.float32)
