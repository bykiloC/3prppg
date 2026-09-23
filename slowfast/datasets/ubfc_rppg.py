#!/usr/bin/env python3

import os
import random
import math

import numpy as np
import slowfast.utils.logging as logging
import torch
import torch.utils.data
from slowfast.quasi_periodic.core import background_anchor_is_active
from slowfast.quasi_periodic.mask import BackgroundMaskCache

from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ubfcrppg(torch.utils.data.Dataset):
    """
    UBFC-rPPG loader for preprocessed npy videos.

    The index format follows the local V-JEPA index:
        npy_path start_frame end_frame label
    """

    def __init__(self, cfg, mode):
        assert mode in ["train", "val", "test"]
        self.cfg = cfg
        self.mode = mode
        self._npy_cache = {}
        self._num_epoch = 0.0
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        if self.cfg.QUASI.ENABLE and self.cfg.QUASI.BACKGROUND_ANCHOR:
            self._background_mask_cache = BackgroundMaskCache()

        index_file = self._get_index_file()
        if not index_file:
            index_file = self.cfg.DATA.PATH_TO_DATA_DIR
        assert index_file and os.path.isfile(index_file), (
            "{} index file not found: {}".format(self.dataset_name, index_file)
        )

        heldout = set(self._get_heldout_subjects())
        self._samples = []
        self._labels = []
        with open(index_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = line.rsplit(maxsplit=3)
                if len(row) != 4:
                    raise RuntimeError(
                        "Expected `npy_path start end label`, got: {}".format(line)
                    )
                path, start, end, label = row
                subject = self._get_subject(path)
                is_heldout = subject in heldout
                if mode == "train" and is_heldout:
                    continue
                if mode in ["val", "test"] and not is_heldout:
                    continue
                self._samples.append(
                    {
                        "path": path,
                        "start": int(start),
                        "end": int(end),
                        "subject": subject,
                    }
                )
                self._labels.append(int(label))

        assert len(self._samples) > 0, (
            "No {} samples for split {} from {}".format(
                self.dataset_name, mode, index_file
            )
        )
        debug_num_samples = self._get_debug_num_samples()
        if debug_num_samples > 0:
            max_samples = debug_num_samples
            self._samples = self._samples[:max_samples]
            self._labels = self._labels[:max_samples]
        logger.info(
            "Constructing {} {} dataloader from {}: {} segments, heldout={}".format(
                self.dataset_name, mode, index_file, len(self._samples), sorted(heldout)
            )
        )

    @property
    def dataset_name(self):
        return "UBFC-rPPG"

    def _get_index_file(self):
        return self.cfg.DATA.UBFC_INDEX_FILE

    def _get_heldout_subjects(self):
        return self.cfg.DATA.UBFC_HELDOUT_SUBJECTS

    def _get_debug_num_samples(self):
        return self.cfg.DATA.UBFC_DEBUG_NUM_SAMPLES

    def _get_subject(self, path):
        return os.path.basename(os.path.dirname(path))

    def _set_epoch_num(self, epoch):
        self._num_epoch = epoch

    def _load_npy(self, path):
        if path not in self._npy_cache:
            self._npy_cache[path] = np.load(path, mmap_mode="r")
        return self._npy_cache[path]

    def _sample_frame_indices(self, start, end):
        num_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        segment_len = max(0, end - start)
        clip_span = (num_frames - 1) * sampling_rate + 1

        if segment_len >= clip_span:
            if self.mode == "train":
                offset = random.randint(0, segment_len - clip_span)
            else:
                offset = max(0, (segment_len - clip_span) // 2)
            first = start + offset
            indices = first + np.arange(num_frames, dtype=np.int64) * sampling_rate
        else:
            indices = np.linspace(start, max(start, end - 1), num_frames)
            indices = indices.astype(np.int64)
        return indices

    def _normalize_channel_layout(self, frames, path):
        if frames.ndim == 4 and frames.shape[-1] in (1, 3, 4):
            pass
        elif frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
            frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            raise RuntimeError(
                "Unsupported channel layout for {}: {}".format(path, frames.shape)
            )
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return frames

    def _load_segment(self, sample):
        video = self._load_npy(sample["path"])
        if video.ndim != 4:
            raise RuntimeError(
                "Unsupported npy shape for {}: {}".format(sample["path"], video.shape)
            )

        video_len = len(video)
        start = max(0, min(sample["start"], video_len))
        end = max(start + 1, min(sample["end"], video_len))
        segment = np.asarray(video[start:end])
        segment = self._normalize_channel_layout(segment, sample["path"])
        return segment, start, end

    def _sample_clip_from_segment(self, segment, start, end):
        indices = self._sample_frame_indices(start, end)
        relative_indices = np.clip(indices - start, 0, len(segment) - 1)
        frames = np.asarray(segment[relative_indices])
        return torch.as_tensor(frames), indices

    def _load_clip(self, sample):
        segment, start, end = self._load_segment(sample)
        return self._sample_clip_from_segment(segment, start, end)

    @staticmethod
    def _interpolate_spatial(tensor, size, mode):
        if mode == "nearest":
            return torch.nn.functional.interpolate(tensor, size=size, mode=mode)
        return torch.nn.functional.interpolate(
            tensor,
            size=size,
            mode=mode,
            align_corners=False,
        )

    def _resize_short_side_pair(
        self,
        frames,
        mask,
        min_scale,
        max_scale,
        inverse_uniform_sampling,
    ):
        if inverse_uniform_sampling:
            size = int(round(1.0 / np.random.uniform(1.0 / max_scale, 1.0 / min_scale)))
        else:
            size = int(round(np.random.uniform(min_scale, max_scale)))

        height = frames.shape[2]
        width = frames.shape[3]
        if (width <= height and width == size) or (height <= width and height == size):
            return frames, mask

        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        frames = self._interpolate_spatial(
            frames, size=(new_height, new_width), mode="bilinear"
        )
        if mask is not None:
            mask = self._interpolate_spatial(
                mask, size=(new_height, new_width), mode="nearest"
            )
        return frames, mask

    def _random_crop_pair(self, frames, mask, crop_size):
        if frames.shape[2] == crop_size and frames.shape[3] == crop_size:
            return frames, mask

        height = frames.shape[2]
        width = frames.shape[3]
        y_offset = int(np.random.randint(0, height - crop_size)) if height > crop_size else 0
        x_offset = int(np.random.randint(0, width - crop_size)) if width > crop_size else 0
        frames = frames[:, :, y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]
        if mask is not None:
            mask = mask[:, :, y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]
        return frames, mask

    def _uniform_crop_pair(self, frames, mask, crop_size, spatial_idx):
        assert spatial_idx in [0, 1, 2]
        height = frames.shape[2]
        width = frames.shape[3]
        y_offset = int(math.ceil((height - crop_size) / 2))
        x_offset = int(math.ceil((width - crop_size) / 2))

        if height > width:
            if spatial_idx == 0:
                y_offset = 0
            elif spatial_idx == 2:
                y_offset = height - crop_size
        else:
            if spatial_idx == 0:
                x_offset = 0
            elif spatial_idx == 2:
                x_offset = width - crop_size

        frames = frames[:, :, y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]
        if mask is not None:
            mask = mask[:, :, y_offset : y_offset + crop_size, x_offset : x_offset + crop_size]
        return frames, mask

    def _horizontal_flip_pair(self, prob, frames, mask):
        if np.random.uniform() < prob:
            frames = frames.flip((-1))
            if mask is not None:
                mask = mask.flip((-1))
        return frames, mask

    def _random_resized_crop_pair(
        self,
        frames,
        mask,
        target_height,
        target_width,
        scale,
        ratio,
    ):
        height = frames.shape[2]
        width = frames.shape[3]
        i, j, h, w = transform._get_param_spatial_crop(scale, ratio, height, width)
        frames = frames[:, :, i : i + h, j : j + w]
        frames = self._interpolate_spatial(
            frames, size=(target_height, target_width), mode="bilinear"
        )
        if mask is not None:
            mask = mask[:, :, i : i + h, j : j + w]
            mask = self._interpolate_spatial(
                mask, size=(target_height, target_width), mode="nearest"
            )
        return frames, mask

    def _random_resized_crop_with_shift_pair(
        self,
        frames,
        mask,
        target_height,
        target_width,
        scale,
        ratio,
    ):
        num_frames = frames.shape[1]
        height = frames.shape[2]
        width = frames.shape[3]

        i, j, h, w = transform._get_param_spatial_crop(scale, ratio, height, width)
        i_, j_, h_, w_ = transform._get_param_spatial_crop(scale, ratio, height, width)
        i_s = [int(x) for x in torch.linspace(i, i_, steps=num_frames).tolist()]
        j_s = [int(x) for x in torch.linspace(j, j_, steps=num_frames).tolist()]
        h_s = [int(x) for x in torch.linspace(h, h_, steps=num_frames).tolist()]
        w_s = [int(x) for x in torch.linspace(w, w_, steps=num_frames).tolist()]

        out_frames = torch.zeros(
            (
                frames.shape[0],
                num_frames,
                target_height,
                target_width,
            ),
            dtype=frames.dtype,
            device=frames.device,
        )
        out_mask = None
        if mask is not None:
            out_mask = torch.zeros(
                (
                    mask.shape[0],
                    num_frames,
                    target_height,
                    target_width,
                ),
                dtype=mask.dtype,
                device=mask.device,
            )

        for t in range(num_frames):
            out_frames[:, t : t + 1] = self._interpolate_spatial(
                frames[
                    :,
                    t : t + 1,
                    i_s[t] : i_s[t] + h_s[t],
                    j_s[t] : j_s[t] + w_s[t],
                ],
                size=(target_height, target_width),
                mode="bilinear",
            )
            if mask is not None:
                out_mask[:, t : t + 1] = self._interpolate_spatial(
                    mask[
                        :,
                        t : t + 1,
                        i_s[t] : i_s[t] + h_s[t],
                        j_s[t] : j_s[t] + w_s[t],
                    ],
                    size=(target_height, target_width),
                    mode="nearest",
                )
        return out_frames, out_mask

    def _spatial_sampling_with_mask(
        self,
        frames,
        mask,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
    ):
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            if aspect_ratio is None and scale is None:
                frames, mask = self._resize_short_side_pair(
                    frames,
                    mask,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    inverse_uniform_sampling=inverse_uniform_sampling,
                )
                frames, mask = self._random_crop_pair(frames, mask, crop_size)
            else:
                if motion_shift:
                    frames, mask = self._random_resized_crop_with_shift_pair(
                        frames,
                        mask,
                        target_height=crop_size,
                        target_width=crop_size,
                        scale=scale,
                        ratio=aspect_ratio,
                    )
                else:
                    frames, mask = self._random_resized_crop_pair(
                        frames,
                        mask,
                        target_height=crop_size,
                        target_width=crop_size,
                        scale=scale,
                        ratio=aspect_ratio,
                    )
            if random_horizontal_flip:
                frames, mask = self._horizontal_flip_pair(0.5, frames, mask)
        else:
            assert len({min_scale, max_scale}) == 1
            frames, mask = self._resize_short_side_pair(
                frames,
                mask,
                min_scale=min_scale,
                max_scale=max_scale,
                inverse_uniform_sampling=False,
            )
            frames, mask = self._uniform_crop_pair(
                frames,
                mask,
                crop_size=crop_size,
                spatial_idx=spatial_idx,
            )
        return frames, mask

    def _transform_clip(self, frames, decode_idx, mask_labels=None):
        mask = None
        if mask_labels is not None:
            if mask_labels.ndim != 3:
                raise ValueError(
                    "Expected T,H,W mask labels, got {}".format(mask_labels.shape)
                )
            if int(mask_labels.shape[0]) != int(frames.shape[0]):
                raise ValueError(
                    "Frame/mask temporal mismatch: frames {} vs mask {}".format(
                        tuple(frames.shape[:3]), tuple(mask_labels.shape)
                    )
                )
            mask = torch.as_tensor(mask_labels.astype(np.float32)).unsqueeze(0)
            if tuple(mask.shape[2:]) != tuple(frames.shape[1:3]):
                mask = self._interpolate_spatial(
                    mask,
                    size=tuple(frames.shape[1:3]),
                    mode="nearest",
                )

        frames = frames.float() / 255.0

        if self.mode == "train" and self.cfg.DATA.SSL_COLOR_JITTER:
            frames = transform.color_jitter_video_ssl(
                frames,
                bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                hue=self.cfg.DATA.SSL_COLOR_HUE,
                p_convert_gray=self.p_convert_gray,
                moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
            )

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        frames = frames.permute(3, 0, 1, 2)

        if self.mode == "train":
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            spatial_sample_index = 1
            min_scale = self.cfg.DATA.TEST_CROP_SIZE
            max_scale = self.cfg.DATA.TEST_CROP_SIZE
            crop_size = self.cfg.DATA.TEST_CROP_SIZE

        relative_scales = (
            None
            if self.mode != "train" or len(self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE) == 0
            else self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE
        )
        relative_aspect = (
            None
            if self.mode != "train" or len(self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE) == 0
            else self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE
        )
        if mask is None:
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                aspect_ratio=relative_aspect,
                scale=relative_scales,
                motion_shift=(
                    self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                    if self.mode == "train"
                    else False
                ),
            )
            return utils.pack_pathway_output(self.cfg, frames)

        frames, mask = self._spatial_sampling_with_mask(
            frames,
            mask,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=(
                self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                if self.mode == "train"
                else False
            ),
        )
        mask_clip = mask.squeeze(0).round().to(torch.uint8).cpu().numpy()
        background_mask = self._background_mask_cache.background_mask_from_clip(
            mask_clip
        )
        return utils.pack_pathway_output(self.cfg, frames), background_mask

    def __getitem__(self, index):
        sample = self._samples[index]
        label = self._labels[index]
        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode == "train" else 1
        )
        use_background_anchor = background_anchor_is_active(
            self.cfg, self._num_epoch
        )

        clips = []
        time_idx = []
        background_masks = []
        segment, start, end = self._load_segment(sample)
        for decode_idx in range(num_decode):
            frames, indices = self._sample_clip_from_segment(segment, start, end)
            if use_background_anchor:
                mask_labels = self._background_mask_cache.load_mask_clip(
                    sample["path"], indices
                )
                clip, background_mask = self._transform_clip(
                    frames,
                    decode_idx,
                    mask_labels=mask_labels,
                )
                background_masks.append(background_mask)
            else:
                clip = self._transform_clip(frames, decode_idx)
            clips.append(clip)
            time_idx.append(indices)

        frames = clips[0] if len(clips) == 1 else clips
        meta = {}
        if use_background_anchor:
            meta["background_masks"] = torch.from_numpy(
                np.stack(background_masks, axis=0)
            )
        return frames, label, index, np.asarray(time_idx), meta

    def __len__(self):
        return len(self._samples)
