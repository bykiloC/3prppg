#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""

import os

from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Video BYOL pretraining options
# -----------------------------------------------------------------------------

_C.CONTRASTIVE = CfgNode()

# temperature used for contrastive losses
_C.CONTRASTIVE.T = 0.07

# output dimension for the loss
_C.CONTRASTIVE.DIM = 128

# legacy checkpoint-compatible queue length. The current BYOL loss does not use
# negatives, but the model still registers this buffer for state_dict stability.
_C.CONTRASTIVE.LENGTH = 239975

# legacy checkpoint-compatible queue length.
_C.CONTRASTIVE.QUEUE_LEN = 65536

# momentum for momentum encoder updates
_C.CONTRASTIVE.MOMENTUM = 0.5

# whether to anneal momentum to value above with cosine schedule
_C.CONTRASTIVE.MOMENTUM_ANNEALING = False

# use an MLP projection with these num layers
_C.CONTRASTIVE.NUM_MLP_LAYERS = 1

# dimension of projection and predictor MLPs
_C.CONTRASTIVE.MLP_DIM = 2048

# use BN in projection/prediction MLP
_C.CONTRASTIVE.BN_MLP = False

# use synchronized BN in projection/prediction MLP
_C.CONTRASTIVE.BN_SYNC_MLP = False

# if non empty, use predictors with depth specified
_C.CONTRASTIVE.PREDICTOR_DEPTHS = []

# Wether to sequentially process multiple clips (=lower mem usage) or batch them
_C.CONTRASTIVE.SEQUENTIAL = False

# Quasi-periodic auxiliary branch. Only experiment-level switches are exposed;
# the validated projection, regularization, and background-mask settings are
# fixed inside slowfast.quasi_periodic.
_C.QUASI = CfgNode()
_C.QUASI.ENABLE = False
_C.QUASI.START_EPOCH = 0
_C.QUASI.BACKGROUND_ANCHOR = False
_C.QUASI.BACKGROUND_ANCHOR_START_EPOCH = 0
_C.QUASI.SD_WEIGHT = 0.1
_C.QUASI.SNR_WEIGHT = 0.1
_C.QUASI.BG_WEIGHT = 1.0


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized. `NUM_SYNC_DEVICES` cannot be larger than number of
# devices per machine; if global sync is desired, set `GLOBAL_SYNC`.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.NUM_SYNC_DEVICES = 1

# Parameter for NaiveSyncBatchNorm. Setting `GLOBAL_SYNC` to True synchronizes
# stats across all devices, across all machines; in this case, `NUM_SYNC_DEVICES`
# must be set to None.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.GLOBAL_SYNC = False

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "ubfcrppg"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Reuse a completed checkpoint as a strict experiment branch point. This mode
# preserves all optimizer state for existing parameters while allowing newly
# enabled modules (currently the quasi-periodic head) to start with empty
# optimizer state.
_C.TRAIN.REUSE_CKPT = False

# Source checkpoint used when TRAIN.REUSE_CKPT is enabled.
_C.TRAIN.CKPT_PATH = ""

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint type. This pruned workspace supports PyTorch checkpoints only.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False

# if True, inflate some params from imagenet model.
_C.TRAIN.CHECKPOINT_IN_INIT = False

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

#  If true, initialize the final conv layer of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_CONV = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Backbone architecture used by the Video BYOL model.
_C.MODEL.ARCH = "slow"

# Output dimension of the BYOL projection head.
_C.MODEL.NUM_CLASSES = 256

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = [
    "c2d",
    "i3d",
    "slow",
    "physnet",
]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False

# If True, detach the final fc layer from the network, by doing so, only the
# final fc layer will be trained.
_C.MODEL.DETACH_FINAL_FC = False

# If True, frozen batch norm stats during training.
_C.MODEL.FROZEN_BN = False

# If True, AllReduce gradients are compressed to fp16
_C.MODEL.FP16_ALLREDUCE = False


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []

# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []

# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# how many samples (=clips) to decode from a single video
_C.DATA.TRAIN_CROP_NUM_TEMPORAL = 1

# color random percentage for grayscale conversion
_C.DATA.COLOR_RND_GRAYSCALE = 0.0

# Apply SSL-based SimCLR / MoCo v1/v2 color augmentations,
#  with params below
_C.DATA.SSL_COLOR_JITTER = False

# color jitter percentage for brightness, contrast, saturation
_C.DATA.SSL_COLOR_BRI_CON_SAT = [0.4, 0.4, 0.4]

# color jitter percentage for hue
_C.DATA.SSL_COLOR_HUE = 0.1

# SimCLR / MoCo v2 augmentations on/off
_C.DATA.SSL_MOCOV2_AUG = False

# SimCLR / MoCo v2 blur augmentation minimum gaussian sigma
_C.DATA.SSL_BLUR_SIGMA_MIN = [0.0, 0.1]

# SimCLR / MoCo v2 blur augmentation maximum gaussian sigma
_C.DATA.SSL_BLUR_SIGMA_MAX = [0.0, 2.0]

# UBFC-rPPG preprocessed npy segment index.
_C.DATA.UBFC_INDEX_FILE = ""
_C.DATA.UBFC_HELDOUT_SUBJECTS = []
_C.DATA.UBFC_DEBUG_NUM_SAMPLES = 0

# PURE preprocessed npy segment index.
_C.DATA.PURE_INDEX_FILE = ""
_C.DATA.PURE_HELDOUT_SUBJECTS = []
_C.DATA.PURE_DEBUG_NUM_SAMPLES = 0

# BUAA-MIHR preprocessed npy segment index.
_C.DATA.BUAA_MIHR_INDEX_FILE = ""
_C.DATA.BUAA_MIHR_HELDOUT_SUBJECTS = []
_C.DATA.BUAA_MIHR_DEBUG_NUM_SAMPLES = 0

# MMPD preprocessed npy segment index.
_C.DATA.MMPD_INDEX_FILE = ""
_C.DATA.MMPD_HELDOUT_SUBJECTS = ["subject31", "subject32", "subject33"]
_C.DATA.MMPD_DEBUG_NUM_SAMPLES = 0

# DEAP preprocessed npy segment index.
_C.DATA.DEAP_INDEX_FILE = ""
_C.DATA.DEAP_HELDOUT_SUBJECTS = ["s19", "s20", "s21", "s22"]
_C.DATA.DEAP_DEBUG_NUM_SAMPLES = 0

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# LARS optimizer
_C.SOLVER.LARS_ON = False

# The layer-wise decay of learning rate. Set to 1. to disable.
_C.SOLVER.LAYER_DECAY = 1.0

# Adam's beta
_C.SOLVER.BETAS = (0.9, 0.999)
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# The name of the current task; e.g. "ssl"/"sl" for (self)supervised learning
_C.TASK = ""

# Number of GPUs to use.
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "."

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Common train data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Add custom config with default values.
custom_config.add_custom_config(_C)


def assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE == "pytorch"
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0
    if cfg.TRAIN.REUSE_CKPT:
        assert cfg.TRAIN.CKPT_PATH, (
            "TRAIN.CKPT_PATH must be set when TRAIN.REUSE_CKPT is enabled"
        )
        assert os.path.isfile(cfg.TRAIN.CKPT_PATH), (
            "TRAIN.CKPT_PATH not found: {}".format(cfg.TRAIN.CKPT_PATH)
        )
        assert not cfg.TRAIN.AUTO_RESUME, (
            "TRAIN.REUSE_CKPT and TRAIN.AUTO_RESUME cannot both be enabled"
        )
        assert not cfg.TRAIN.CHECKPOINT_FILE_PATH, (
            "TRAIN.REUSE_CKPT and TRAIN.CHECKPOINT_FILE_PATH cannot both be set"
        )
        assert not cfg.TRAIN.CHECKPOINT_EPOCH_RESET, (
            "TRAIN.REUSE_CKPT must preserve the source checkpoint epoch"
        )

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    if cfg.QUASI.BACKGROUND_ANCHOR:
        assert cfg.QUASI.ENABLE, "QUASI.BACKGROUND_ANCHOR requires QUASI.ENABLE"
        assert cfg.QUASI.BACKGROUND_ANCHOR_START_EPOCH >= 0, (
            "QUASI.BACKGROUND_ANCHOR_START_EPOCH must be non-negative"
        )
        assert isinstance(cfg.QUASI.BACKGROUND_ANCHOR_START_EPOCH, int), (
            "QUASI.BACKGROUND_ANCHOR_START_EPOCH must be an integer epoch"
        )
    if cfg.QUASI.ENABLE:
        assert cfg.CONTRASTIVE.SEQUENTIAL, (
            "The quasi-periodic branch is wired to the current sequential "
            "two-view Video BYOL path."
        )
        assert cfg.DATA.TRAIN_CROP_NUM_TEMPORAL == 2, (
            "The quasi-periodic branch expects exactly two BYOL views."
        )
        assert cfg.MODEL.ARCH == "slow", (
            "The validated quasi-periodic branch is attached to SlowR50 S4."
        )

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
