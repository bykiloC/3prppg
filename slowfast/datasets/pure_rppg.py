#!/usr/bin/env python3

import os

from .build import DATASET_REGISTRY
from .ubfc_rppg import Ubfcrppg


@DATASET_REGISTRY.register()
class Pure(Ubfcrppg):
    """
    PURE loader for preprocessed npy videos.

    The index format is shared with UBFC-rPPG:
        npy_path start_frame end_frame label

    PURE npy files are named like 01-01.npy, where the subject id is the first
    field before the dash.
    """

    @property
    def dataset_name(self):
        return "PURE"

    def _get_index_file(self):
        return self.cfg.DATA.PURE_INDEX_FILE

    def _get_heldout_subjects(self):
        return self.cfg.DATA.PURE_HELDOUT_SUBJECTS

    def _get_debug_num_samples(self):
        return self.cfg.DATA.PURE_DEBUG_NUM_SAMPLES

    def _get_subject(self, path):
        return os.path.basename(path).split("-", 1)[0]
