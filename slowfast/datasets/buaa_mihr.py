#!/usr/bin/env python3

import os

from .build import DATASET_REGISTRY
from .ubfc_rppg import Ubfcrppg


@DATASET_REGISTRY.register()
class Buaa_mihr(Ubfcrppg):
    """
    BUAA-MIHR loader for preprocessed npy videos.

    The index format is shared with UBFC-rPPG and PURE:
        npy_path start_frame end_frame label

    BUAA-MIHR npy files are stored under subject folders such as:
        Sub 01/lux 1.0/*_fps30.npy
    """

    @property
    def dataset_name(self):
        return "BUAA-MIHR"

    def _get_index_file(self):
        return self.cfg.DATA.BUAA_MIHR_INDEX_FILE

    def _get_heldout_subjects(self):
        return self.cfg.DATA.BUAA_MIHR_HELDOUT_SUBJECTS

    def _get_debug_num_samples(self):
        return self.cfg.DATA.BUAA_MIHR_DEBUG_NUM_SAMPLES

    def _get_subject(self, path):
        return os.path.basename(os.path.dirname(os.path.dirname(path)))
