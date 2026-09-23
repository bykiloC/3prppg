#!/usr/bin/env python3

"""Quasi-periodic projection and regularization for Video BYOL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTION_DIM = 256
TEMPORAL_KERNEL_SIZE = 9
TEMPORAL_STRIDE = 1
REVIN_EPS = 1e-5
SAMPLE_RATE_HZ = 15.0
SNR_FREQ_LOW_HZ = 0.5
SNR_FREQ_HIGH_HZ = 3.0
BACKGROUND_CORR_THRESHOLD = 0.25


def background_anchor_is_active(cfg, epoch_exact):
    """Return whether the background-anchor loss is active at this epoch."""
    if not bool(cfg.QUASI.ENABLE) or not bool(cfg.QUASI.BACKGROUND_ANCHOR):
        return False
    start_epoch = float(cfg.QUASI.BACKGROUND_ANCHOR_START_EPOCH)
    if start_epoch < 0.0:
        raise ValueError(
            "QUASI.BACKGROUND_ANCHOR_START_EPOCH must be non-negative."
        )
    quasi_start_epoch = float(cfg.QUASI.START_EPOCH)
    effective_start_epoch = max(start_epoch, quasi_start_epoch)
    if epoch_exact is None:
        return effective_start_epoch <= 0.0
    return float(epoch_exact) >= effective_start_epoch


class QuasiProjectionHead(nn.Module):
    """BCT+CoS projection head used by the validated minimal experiment."""

    def __init__(self, in_channels):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv1d(in_channels, PROJECTION_DIM, kernel_size=1, bias=False),
            nn.BatchNorm1d(PROJECTION_DIM),
            nn.Tanh(),
        )
        self.cos_conv = nn.Conv1d(
            PROJECTION_DIM,
            PROJECTION_DIM,
            kernel_size=TEMPORAL_KERNEL_SIZE,
            stride=TEMPORAL_STRIDE,
            padding=0,
        )
        self.apply(self._weight_init)

    @torch.no_grad()
    def _weight_init(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected B,C,T input, got {}".format(tuple(x.shape)))
        x = self.adapter(x)
        return self.cos_conv(x)


def temporal_revin(feat):
    """Temporal RevIN for B,C,T features, with affine transformation disabled."""
    if feat.ndim != 3:
        raise ValueError("Expected B,C,T feature, got {}".format(tuple(feat.shape)))
    mean = feat.mean(dim=-1, keepdim=True).detach()
    std = torch.sqrt(
        torch.var(feat, dim=-1, keepdim=True, unbiased=False) + REVIN_EPS
    ).detach()
    return (feat - mean) / std


def global_temporal_features(spatial):
    if spatial.ndim != 5:
        raise ValueError(
            "Expected B,C,T,H,W S4 feature, got {}".format(tuple(spatial.shape))
        )
    return temporal_revin(spatial.mean(dim=(-1, -2)))


def masked_gap(spatial, mask):
    if spatial.ndim != 5:
        raise ValueError(
            "Expected B,C,T,H,W S4 feature, got {}".format(tuple(spatial.shape))
        )
    if mask.ndim != 4:
        raise ValueError("Expected B,T,H,W mask, got {}".format(tuple(mask.shape)))
    if tuple(spatial.shape[2:]) != tuple(mask.shape[1:]):
        raise ValueError(
            "S4/mask shape mismatch: spatial {} vs mask {}".format(
                tuple(spatial.shape[2:]), tuple(mask.shape[1:])
            )
        )
    weights = mask.to(device=spatial.device, dtype=spatial.dtype).unsqueeze(1)
    denom = weights.sum(dim=(-1, -2)).clamp_min(1.0)
    return (spatial * weights).sum(dim=(-1, -2)) / denom


def background_temporal_features(spatial, mask):
    return temporal_revin(masked_gap(spatial, mask))


def temporal_ssm(z):
    z_norm = F.normalize(z, p=2, dim=1, eps=1e-6)
    return torch.einsum("bdt,bdu->btu", z_norm, z_norm)


def sd_regularizer(ssm):
    lag_losses = []
    time_len = ssm.shape[-1]
    for lag in range(1, time_len - 1):
        diag = torch.diagonal(ssm, offset=lag, dim1=1, dim2=2)
        if diag.shape[-1] < 2:
            continue
        scaled = diag * (float(diag.shape[-1]) / 20.0)
        lag_losses.append(torch.std(scaled, dim=-1))
    if not lag_losses:
        return ssm.new_tensor(0.0)
    return torch.stack(lag_losses, dim=-1).mean()


def lag_wave_from_ssm(ssm, include_zero=False):
    waves = []
    time_len = ssm.shape[-1]
    start = 0 if include_zero else 1
    for lag in range(start, time_len):
        diag = torch.diagonal(ssm, offset=lag, dim1=1, dim2=2)
        waves.append(diag.mean(dim=-1))
    return torch.stack(waves, dim=-1)


def snr_regularizer(ssm):
    wave = lag_wave_from_ssm(ssm, include_zero=True).float()
    wave_psd = torch.fft.fft(wave, dim=-1).abs()
    wave_len = int(wave_psd.shape[-1])
    half_len = max(1, wave_len // 2)
    wave_psd = wave_psd[:, :half_len]
    wave_psd = wave_psd / wave_psd.amax(dim=-1, keepdim=True).clamp_min(1e-6)

    lf = int(SNR_FREQ_LOW_HZ * wave_len // SAMPLE_RATE_HZ)
    hf = int(SNR_FREQ_HIGH_HZ * wave_len // SAMPLE_RATE_HZ)
    lf = max(0, min(lf, half_len - 1))
    hf = max(1, min(hf, half_len))
    hf = max(hf, lf + 1)

    total_energy = wave_psd.sum(dim=-1)
    signal_energy = wave_psd[:, lf:hf].sum(dim=-1)
    noise_energy = (total_energy - signal_energy).clamp_min(0.0)
    loss_per_sample = noise_energy / (total_energy + 1e-6)
    return loss_per_sample.mean()


def quasi_regularization(spatial, head, sd_weight=1.0, snr_weight=1.0):
    """Compute the joint SD+SNR quasi-periodic regularization."""
    feat = global_temporal_features(spatial)
    z = head(feat)
    ssm = temporal_ssm(z)
    loss_sd = sd_regularizer(ssm)
    loss_snr = snr_regularizer(ssm)
    loss_reg = float(sd_weight) * loss_sd + float(snr_weight) * loss_snr
    lag_wave = lag_wave_from_ssm(ssm, include_zero=False)
    return loss_reg, loss_sd, loss_snr, lag_wave


def background_lag_wave(spatial, mask, head):
    feat = background_temporal_features(spatial, mask)
    z = head(feat)
    return lag_wave_from_ssm(temporal_ssm(z), include_zero=False)


def per_sample_pearson_corr(x, y):
    if x.shape != y.shape:
        raise ValueError(
            "PCC inputs must have same shape, got {} and {}".format(
                tuple(x.shape), tuple(y.shape)
            )
        )
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    numerator = (x * y).sum(dim=-1)
    denominator = x.pow(2).sum(dim=-1).sqrt() * y.pow(2).sum(dim=-1).sqrt()
    return numerator / denominator.clamp_min(1e-6)


def background_anchor_loss(global_lag_wave, teacher_background_lag_wave):
    teacher_background_lag_wave = teacher_background_lag_wave.detach()
    corr = per_sample_pearson_corr(global_lag_wave, teacher_background_lag_wave)
    active = corr > BACKGROUND_CORR_THRESHOLD
    loss = torch.relu(corr - BACKGROUND_CORR_THRESHOLD).mean()
    return loss, corr.mean(), active.float().mean()
