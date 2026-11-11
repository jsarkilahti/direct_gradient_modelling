"""Manage reproducible random-number generation and experiment seed configurations."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any
import torch

def make_torch_generator(
    seed: int | None,
    device: str | torch.device = "cpu",
) -> torch.Generator:
    """
    Create a deterministic torch.Generator.

    Parameters
    ----------
    seed
        Seed used to initialize the generator. If None, an unseeded generator is
        returned.

    device
        Device associated with the generator.

    Returns
    -------
    torch.Generator
        Generator suitable for reproducible sampling in simulate_data.py.
    """

    generator = torch.Generator(device=device)

    if seed is not None:
        generator.manual_seed(int(seed))

    return generator

def add_white_noise_at_snr(
    signal: torch.Tensor,
    snr_db: float,
    *,
    rng: torch.Generator,
    center_noise: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Add white Gaussian noise to a signal at a requested per-trajectory SNR.

    Parameters
    ----------
    signal
        Clean signal [B, T, C].

    snr_db
        Requested signal-to-noise ratio in dB.

    rng
        Torch generator for reproducible noise.

    center_noise
        If True, subtract each trajectory/channel noise mean before scaling.

    eps
        Small numerical stabilizer.

    Returns
    -------
    noisy_signal, metadata
        Corrupted signal and metadata including requested and realised SNR.

    Notes
    -----
    The noise is normalized and scaled per trajectory/channel so that the
    realised SNR is as close as possible to the requested value for each series.
    """

    if signal.dim() != 3:
        raise ValueError(f"Expected signal with shape [B, T, C], got {tuple(signal.shape)}.")

    clean = signal.clone()
    signal_power = clean.pow(2).mean(dim=1, keepdim=True)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    target_noise_power = signal_power / snr_linear
    target_noise_std = torch.sqrt(target_noise_power)

    noise = torch.randn(
        clean.shape,
        dtype=clean.dtype,
        device=clean.device,
        generator=rng,
    )

    if center_noise:
        noise = noise - noise.mean(dim=1, keepdim=True)

    noise_rms = torch.sqrt(
        noise.pow(2).mean(dim=1, keepdim=True)
    )
    noise = noise / (noise_rms + eps)
    noise = noise * target_noise_std

    noisy = clean + noise

    realised = compute_snr_db(clean, noise, eps=eps)

    metadata = {
        "requested_snr_db": float(snr_db),
        "realised_snr_db_per_series_channel": realised.detach().cpu().numpy().tolist(),
        "realised_snr_db_mean": float(torch.mean(realised).detach().cpu()),
        "signal_power_per_series_channel": signal_power.squeeze(1).detach().cpu().numpy().tolist(),
        "noise_power_per_series_channel": noise.pow(2).mean(dim=1).detach().cpu().numpy().tolist(),
        "center_noise": bool(center_noise),
    }

    return noisy, metadata

def compute_snr_db(
    signal: torch.Tensor,
    noise: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute per-trajectory/channel SNR in dB.

    Parameters
    ----------
    signal
        Clean signal [B, T, C].

    noise
        Additive noise [B, T, C].

    Returns
    -------
    torch.Tensor
        SNR values [B, C].
    """

    if signal.shape != noise.shape:
        raise ValueError(
            f"signal and noise must have identical shapes, got {tuple(signal.shape)} and {tuple(noise.shape)}."
        )

    signal_power = signal.pow(2).mean(dim=1)
    noise_power = noise.pow(2).mean(dim=1)

    return 10.0 * torch.log10((signal_power + eps) / (noise_power + eps))

def _serializable_dataclass(value):
    """Handle serializable dataclass for the experiment workflow."""
    if is_dataclass(value):
        return asdict(value)
    return value

def apply_noise_config(
    controls: torch.Tensor,
    outputs: torch.Tensor,
    noise_config: NoiseConfig,
    *,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Apply configured control and output noise to clean trajectories.

    Parameters
    ----------
    controls
        Clean controls [B, T, input_dim].

    outputs
        Clean outputs [B, T, output_dim].

    noise_config
        Noise configuration.

    rng
        Shared generator. Calls are deterministic and ordered.

    Returns
    -------
    noisy_controls, noisy_outputs, metadata
        Possibly corrupted signals and detailed noise/SNR metadata.
    """

    noisy_controls = controls.clone()
    noisy_outputs = outputs.clone()
    metadata: dict[str, Any] = {
        "noise_config": _serializable_dataclass(noise_config),
        "control_noise": None,
        "output_noise": None,
    }

    if noise_config.add_control_noise:
        if noise_config.control_snr_db is None:
            raise ValueError(
                "control_snr_db must be provided when add_control_noise=True."
            )

        noisy_controls, control_metadata = add_white_noise_at_snr(
            noisy_controls,
            float(noise_config.control_snr_db),
            rng=rng,
            center_noise=noise_config.center_noise,
            eps=noise_config.eps,
        )
        metadata["control_noise"] = control_metadata

    if noise_config.add_output_noise:
        if noise_config.output_snr_db is None:
            raise ValueError(
                "output_snr_db must be provided when add_output_noise=True."
            )

        noisy_outputs, output_metadata = add_white_noise_at_snr(
            noisy_outputs,
            float(noise_config.output_snr_db),
            rng=rng,
            center_noise=noise_config.center_noise,
            eps=noise_config.eps,
        )
        metadata["output_noise"] = output_metadata

    return noisy_controls, noisy_outputs, metadata

