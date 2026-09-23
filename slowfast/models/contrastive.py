#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math

import torch
import torch.nn as nn
from slowfast.models.physnet import PhysNet
from slowfast.models.video_model_builder import ResNet, SlowFast
from slowfast.quasi_periodic.core import (
    QuasiProjectionHead,
    background_anchor_is_active,
    background_anchor_loss,
    background_lag_wave,
    quasi_regularization,
)

from .build import MODEL_REGISTRY

# Supported backbone types. This file is pruned to the Video BYOL baseline, but
# the backbone library is intentionally left intact.
_MODEL_TYPES = {
    "slowfast": SlowFast,
    "slow": ResNet,
    "c2d": ResNet,
    "i3d": ResNet,
    "physnet": PhysNet,
}


def quasi_is_active(cfg, epoch_exact):
    """Return whether the curriculum-gated quasi branch is active."""
    if not bool(cfg.QUASI.ENABLE):
        return False
    start_epoch = float(cfg.QUASI.START_EPOCH)
    if start_epoch < 0.0:
        raise ValueError("QUASI.START_EPOCH must be non-negative.")
    if epoch_exact is None:
        return start_epoch <= 0.0
    return float(epoch_exact) >= start_epoch


@MODEL_REGISTRY.register()
class ContrastiveModel(nn.Module):
    """
    Video BYOL pretraining model.

    The public class name is kept as ``ContrastiveModel`` so existing configs
    and checkpoints keep the same entrypoint.
    """

    def __init__(self, cfg):
        super(ContrastiveModel, self).__init__()
        self.backbone = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
        self.backbone_hist = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
        for p in self.backbone_hist.parameters():
            p.requires_grad = False

        self.T = cfg.CONTRASTIVE.T
        self.dim = cfg.CONTRASTIVE.DIM
        self.length = cfg.CONTRASTIVE.LENGTH
        self.k = cfg.CONTRASTIVE.QUEUE_LEN
        self.mmt = cfg.CONTRASTIVE.MOMENTUM
        self.momentum_annealing = cfg.CONTRASTIVE.MOMENTUM_ANNEALING
        self.cfg = cfg
        self.num_gpus = cfg.NUM_GPUS
        self.l2_norm = Normalize()

        self.quasi_enabled = bool(cfg.QUASI.ENABLE)
        self.quasi_start_epoch = float(cfg.QUASI.START_EPOCH)
        if self.quasi_start_epoch < 0.0:
            raise ValueError("QUASI.START_EPOCH must be non-negative.")
        self.background_anchor_enabled = bool(cfg.QUASI.BACKGROUND_ANCHOR)
        if self.quasi_enabled:
            s4_channels = cfg.RESNET.WIDTH_PER_GROUP * 16
            self.quasi_projection_head = QuasiProjectionHead(
                in_channels=s4_channels,
            )
            self.quasi_projection_head_hist = QuasiProjectionHead(
                in_channels=s4_channels,
            )
            self.quasi_projection_head_hist.load_state_dict(
                self.quasi_projection_head.state_dict()
            )
            for p in self.quasi_projection_head_hist.parameters():
                p.requires_grad = False

        # These legacy buffers are not used by BYOL loss computation, but keeping
        # them preserves the checkpoint key set and the initialization RNG stream.
        self.register_buffer("ptr", torch.tensor([0]))
        self.ptr.requires_grad = False
        stdv = 1.0 / math.sqrt(self.dim / 3)
        self.register_buffer(
            "queue_x",
            torch.rand(self.k, self.dim).mul_(2 * stdv).add_(-stdv),
        )
        self.register_buffer("iter", torch.zeros([1], dtype=torch.long))

    @torch.no_grad()
    def _update_history(self, quasi_active):
        iter_num = int(self.iter)
        m = self.mmt
        params = {}
        for name, p in self.backbone.named_parameters():
            params[name] = p

        if iter_num == 0:
            for name, p in self.backbone_hist.named_parameters():
                p.data.copy_(params[name].data)

        for name, p in self.backbone_hist.named_parameters():
            p.data = params[name].data * (1.0 - m) + p.data * m

        if self.quasi_enabled and quasi_active:
            quasi_params = dict(self.quasi_projection_head.named_parameters())
            if iter_num == 0:
                for name, p in self.quasi_projection_head_hist.named_parameters():
                    p.data.copy_(quasi_params[name].data)
            for name, p in self.quasi_projection_head_hist.named_parameters():
                p.data = quasi_params[name].data * (1.0 - m) + p.data * m

    @torch.no_grad()
    def batch_clips(self, clips):
        clips_batched = [None] * len(clips[0])
        for i, clip in enumerate(clips):
            for j, view in enumerate(clip):
                if i == 0:
                    clips_batched[j] = view
                else:
                    clips_batched[j] = torch.cat([clips_batched[j], view], dim=0)
                del view
        return clips_batched

    @torch.no_grad()
    def compute_key_feat(
        self,
        clips_k,
        compute_predictor_keys=False,
        batched_inference=True,
        background_masks=None,
        quasi_active=None,
        background_anchor_active=None,
    ):
        assert self.training
        if quasi_active is None:
            # Keep direct legacy callers equivalent to the pre-curriculum path.
            quasi_active = self.quasi_enabled
        quasi_active = bool(quasi_active)
        if background_anchor_active is None:
            background_anchor_active = background_anchor_is_active(self.cfg, None)
        background_anchor_active = bool(background_anchor_active)
        self._update_history(quasi_active=quasi_active)
        self.iter += 1

        n_clips = len(clips_k)
        bsz = clips_k[0][0].shape[0]
        if n_clips * bsz * clips_k[0][0].numel() > 4 * 64 * 3 * 8 * 224 * 224:
            batched_inference = False
        assert n_clips > 0

        if background_anchor_active:
            if compute_predictor_keys:
                raise NotImplementedError(
                    "Teacher background anchors are only used with BYOL projector keys."
                )
            if background_masks is None:
                raise ValueError(
                    "QUASI.BACKGROUND_ANCHOR requires background masks from the dataset."
                )
            if background_masks.ndim != 5 or background_masks.shape[1] != n_clips:
                raise ValueError(
                    "Expected background masks B,V,T,H,W for {} views, got {}".format(
                        n_clips, tuple(background_masks.shape)
                    )
                )
            # Keep teacher views separate so each S4 feature remains aligned with
            # its own precomputed background mask.
            batched_inference = False

        if batched_inference and all(
            [
                clips_k[i][j].shape[1:] == clips_k[0][j].shape[1:]
                for i in range(len(clips_k))
                for j in range(len(clips_k[i]))
            ]
        ):
            clips_k = [self.batch_clips(clips_k)]
            batched = True
        else:
            batched = False

        keys, pred_keys = [], []
        background_anchors = []
        for k in range(0, len(clips_k)):
            clip_k = clips_k[k]
            with torch.no_grad():
                if background_anchor_active:
                    hist_feat, s4_hist = self.backbone_hist(clip_k, return_s4=True)
                else:
                    hist_feat = self.backbone_hist(clip_k)
                    s4_hist = None

                if isinstance(hist_feat, list):
                    hist_time = hist_feat[1:]
                    hist_feat = hist_feat[0]
                    if compute_predictor_keys:
                        tks = []
                        for tk in hist_time:
                            tks.append(self.l2_norm(tk))
                        pred_keys.append(tks)
                x_hist = self.l2_norm(hist_feat)
                keys.append(x_hist)

                if background_anchor_active:
                    mask = background_masks[:, k]
                    bg_wave = background_lag_wave(
                        s4_hist,
                        mask,
                        self.quasi_projection_head_hist,
                    )
                    background_anchors.append(bg_wave.detach())

        if batched:
            assert len(keys) == 1, "batched input uses single clip"
            batched_key = keys[0]
            if compute_predictor_keys:
                batched_pred_key = pred_keys[0]
            keys, pred_keys = [], []
            for k in range(0, n_clips):
                keys.append(batched_key[k * bsz : (k + 1) * bsz])
                if compute_predictor_keys:
                    pred_keys.append(batched_pred_key[k * bsz : (k + 1) * bsz])

        if background_anchor_active:
            return keys, background_anchors
        if compute_predictor_keys:
            return keys, pred_keys
        return keys

    def sim_loss(self, q, k):
        similarity = torch.einsum("nc,nc->n", [q, k])
        similarity /= self.T
        loss = -similarity.mean()
        return loss

    @torch.no_grad()
    def momentum_anneal_cosine(self, epoch_exact):
        self.mmt = (
            1
            - (1 - self.cfg.CONTRASTIVE.MOMENTUM)
            * (math.cos(math.pi * epoch_exact / self.cfg.SOLVER.MAX_EPOCH) + 1.0)
            * 0.5
        )

    def forward(
        self,
        clips,
        index=None,
        time=None,
        epoch_exact=None,
        keys=None,
        background_anchors=None,
    ):
        quasi_active = quasi_is_active(self.cfg, epoch_exact)
        background_anchor_active = background_anchor_is_active(
            self.cfg, epoch_exact
        )
        if epoch_exact is not None and self.momentum_annealing:
            self.momentum_anneal_cosine(epoch_exact)

        clips_key = [None] * len(clips)
        for i, clip in enumerate(clips):
            p = []
            for path in clip:
                p.append(path)
            clips_key[i] = p

        batch_clips = False
        if isinstance(clips[0], list):
            n_clips = len(clips)
            if batch_clips and n_clips > 1:
                clips_batched = self.batch_clips(clips)
                clips_key = [clips_batched]
                clip_q = clips_batched
            else:
                clip_q = clips[0]
        else:
            clip_q = clips

        if quasi_active:
            feat_q, s4_q = self.backbone(clip_q, return_s4=True)
        else:
            feat_q = self.backbone(clip_q)
            s4_q = None
        predictors = []
        if isinstance(feat_q, list):
            predictors = feat_q[1:]
            feat_q = feat_q[0]
            predictors = [self.l2_norm(feat) for feat in predictors]
        else:
            raise NotImplementedError("BYOL: predictor is missing")
        assert len(predictors) == 1

        if index is None:
            return feat_q

        if not self.training:
            return self.l2_norm(feat_q)

        if keys is None:
            keys = self.compute_key_feat(
                clips_key,
                compute_predictor_keys=False,
                quasi_active=quasi_active,
                background_anchor_active=background_anchor_active,
            )

        if self.cfg.CONTRASTIVE.SEQUENTIAL:
            loss_reg = self.sim_loss(predictors[0], keys[0])
            for i in range(1, len(keys)):
                loss_reg += self.sim_loss(predictors[0], keys[i])
            loss_reg /= len(keys)
        else:
            if batch_clips:
                bs = predictors[0].shape[0] // 2
                loss_reg = self.sim_loss(
                    predictors[0][:bs, :], keys[0][bs:, :]
                ) + self.sim_loss(predictors[0][bs:, :], keys[0][:bs, :])
                del clips_batched[0]
            else:
                loss_q1 = self.sim_loss(predictors[0], keys[1])
                assert len(clips) == 2
                clip_q2 = clips[1]
                feat_q2 = self.backbone(clip_q2)
                predictors2 = feat_q2[1:]
                predictors2 = [self.l2_norm(feat) for feat in predictors2]
                assert len(predictors2) == 1

                loss_q2 = self.sim_loss(predictors2[0], keys[0])
                loss_reg = loss_q1 + loss_q2

        loss_components = None
        if self.quasi_enabled:
            zero = loss_reg.detach().new_zeros(())
            loss_components = {
                "quasi_active": loss_reg.detach().new_tensor(
                    float(quasi_active)
                ),
                "background_anchor_active": loss_reg.detach().new_tensor(
                    float(background_anchor_active)
                ),
                "quasi_start_epoch": loss_reg.detach().new_tensor(
                    self.quasi_start_epoch
                ),
                "loss_byol": loss_reg.detach(),
                "loss_sd": zero,
                "loss_snr": zero,
                "loss_quasi_reg": zero,
                "loss_sd_weighted": zero,
                "loss_snr_weighted": zero,
                "loss_bg_anchor": zero,
                "loss_bg_anchor_weighted": zero,
            }

        if quasi_active:
            (
                loss_quasi_reg,
                loss_sd,
                loss_snr,
                global_lag_wave,
            ) = quasi_regularization(
                s4_q,
                self.quasi_projection_head,
                sd_weight=self.cfg.QUASI.SD_WEIGHT,
                snr_weight=self.cfg.QUASI.SNR_WEIGHT,
            )
            loss_total = loss_reg + loss_quasi_reg

            loss_components.update(
                {
                    "loss_sd": loss_sd.detach(),
                    "loss_snr": loss_snr.detach(),
                    "loss_quasi_reg": loss_quasi_reg.detach(),
                    "loss_sd_weighted": (
                        loss_sd * float(self.cfg.QUASI.SD_WEIGHT)
                    ).detach(),
                    "loss_snr_weighted": (
                        loss_snr * float(self.cfg.QUASI.SNR_WEIGHT)
                    ).detach(),
                }
            )

            if background_anchor_active:
                if not background_anchors:
                    raise ValueError(
                        "Teacher background anchors are required when "
                        "QUASI.BACKGROUND_ANCHOR is enabled."
                    )
                anchor_losses = []
                anchor_corrs = []
                anchor_active = []
                for anchor in background_anchors:
                    loss_anchor, corr_mean, active_fraction = background_anchor_loss(
                        global_lag_wave,
                        anchor,
                    )
                    anchor_losses.append(loss_anchor)
                    anchor_corrs.append(corr_mean)
                    anchor_active.append(active_fraction)
                loss_anchor = torch.stack(anchor_losses).mean()
                corr_mean = torch.stack(anchor_corrs).mean()
                active_fraction = torch.stack(anchor_active).mean()
                loss_anchor_weighted = (
                    loss_anchor * float(self.cfg.QUASI.BG_WEIGHT)
                )
                loss_total = loss_total + loss_anchor_weighted
                loss_components.update(
                    {
                        "loss_bg_anchor": loss_anchor.detach(),
                        "loss_bg_anchor_weighted": loss_anchor_weighted.detach(),
                        "bg_pcc": corr_mean.detach(),
                        "bg_hinge_active": active_fraction.detach(),
                    }
                )

            loss_reg = loss_total

        dummy_logits = torch.cat(
            (
                9999.0 * torch.ones((len(index), 1), dtype=torch.float).cuda(),
                torch.zeros((len(index), self.k), dtype=torch.float).cuda(),
            ),
            dim=1,
        )
        if self.quasi_enabled:
            return dummy_logits, loss_reg, loss_components
        return dummy_logits, loss_reg


class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.dim = dim
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


def contrastive_parameter_surgery(model, cfg, epoch_exact, cur_iter):
    return model, True


def contrastive_forward(model, cfg, inputs, index, time, epoch_exact, scaler, meta=None):
    quasi_stats = {}
    quasi_active = quasi_is_active(cfg, epoch_exact)
    background_anchor_active = background_anchor_is_active(cfg, epoch_exact)
    if cfg.CONTRASTIVE.SEQUENTIAL:
        perform_backward = False
        mdl = getattr(model, "module", model)
        background_masks = None
        if background_anchor_active:
            if meta is None or "background_masks" not in meta:
                raise ValueError(
                    "QUASI.BACKGROUND_ANCHOR requires meta['background_masks']."
                )
            background_masks = meta["background_masks"]

        key_output = mdl.compute_key_feat(
            inputs,
            compute_predictor_keys=False,
            batched_inference=True if len(inputs) < 2 else False,
            background_masks=background_masks,
            quasi_active=quasi_active,
            background_anchor_active=background_anchor_active,
        )
        if background_anchor_active:
            keys, background_anchors = key_output
        else:
            keys = key_output
            background_anchors = None

        for k, vid in enumerate(inputs):
            other_keys = keys[:k] + keys[k + 1 :]
            other_background_anchors = (
                background_anchors[:k] + background_anchors[k + 1 :]
                if background_anchors is not None
                else None
            )
            time_cur = torch.cat(
                [
                    time[:, k : k + 1, :],
                    time[:, :k, :],
                    time[:, k + 1 :, :],
                ],
                1,
            )
            output = model(
                [vid],
                index,
                time_cur,
                epoch_exact,
                keys=other_keys,
                background_anchors=other_background_anchors,
            )
            if cfg.QUASI.ENABLE:
                lgt_k, loss_k, stats_k = output
                for name, value in stats_k.items():
                    quasi_stats[name] = quasi_stats.get(name, 0.0) + value
            else:
                lgt_k, loss_k = output

            scaler.scale(loss_k).backward()
            if k == 0:
                preds, partial_loss = lgt_k, loss_k.detach()
            else:
                preds = torch.cat([preds, lgt_k], dim=0)
                partial_loss += loss_k.detach()

        partial_loss /= len(inputs) * 2.0
        if quasi_stats:
            for name in quasi_stats:
                quasi_stats[name] = quasi_stats[name] / float(len(inputs))
    else:
        perform_backward = True
        output = model(inputs, index, time, epoch_exact, keys=None)
        if cfg.QUASI.ENABLE:
            preds, partial_loss, quasi_stats = output
        else:
            preds, partial_loss = output
    return model, preds, partial_loss, perform_backward, quasi_stats
