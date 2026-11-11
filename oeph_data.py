"""Load or simulate MSD and cascaded-tanks time series; prepare per-method preprocessing and raw-data caches."""
from __future__ import annotations
import csv
import hashlib
import io
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from oeph_definitions import (TimeSeriesDataset, SplitDataset, MSD2Config, InitialConditionConfig, ControlConfig, NoiseConfig, SimulationConfig, ExperimentConfig)
from oeph_physics import MSD2
from oeph_random import make_torch_generator, apply_noise_config

def _control_interval_lengths_samples(
    *,
    sampling_time: float,
    durations: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert the configured control-region duration bank to sample counts."""

    if not np.isfinite(float(sampling_time)) or float(sampling_time) <= 0.0:
        raise ValueError("sampling_time must be finite and positive.")
    if not durations:
        raise ValueError("zero_interval_durations must contain at least one duration.")

    lengths: list[int] = []
    for duration in durations:
        value = float(duration)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "zero_interval_durations must contain only finite positive values."
            )
        lengths.append(max(1, int(round(value / float(sampling_time)))))

    return tuple(lengths)

def _mask_intervals(
    mask: torch.Tensor,
    *,
    value: bool,
) -> list[tuple[int, int]]:
    """Return contiguous [start, end) intervals where ``mask == value``."""

    array = mask.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
    intervals: list[tuple[int, int]] = []
    start: int | None = None

    for index, item in enumerate(array):
        if bool(item) == bool(value) and start is None:
            start = index
        elif bool(item) != bool(value) and start is not None:
            intervals.append((start, index))
            start = None

    if start is not None:
        intervals.append((start, int(array.size)))

    return intervals

def _correct_control_zero_count(
    active_mask: torch.Tensor,
    *,
    target_zero_count: int,
    interval_lengths: tuple[int, ...],
    rng: torch.Generator,
) -> torch.Tensor:
    """
    Correct the realised ZERO fraction by random interval replacement.

    The proposal process already generates contiguous ZERO/ACTIVE regions from
    ``interval_lengths``.  This correction preserves that block structure:

    - if more ZERO samples are required, a randomly selected subinterval of an
      ACTIVE region is replaced by ZERO;
    - if fewer ZERO samples are required, a randomly selected subinterval of a
      ZERO region is replaced by ACTIVE.

    Replacement lengths are drawn uniformly from ``interval_lengths`` and are
    clipped only by the remaining sample-count error and the selected source
    interval.  Whenever possible, the replacement is placed strictly inside
    the source interval so that it does not merge with neighbouring intervals.

    The procedure terminates when the requested ZERO sample count is attained
    exactly.
    """

    length = int(active_mask.numel())
    target = int(target_zero_count)

    if not 0 <= target <= length:
        raise ValueError("target_zero_count must lie in [0, series_length].")
    if not interval_lengths:
        raise ValueError("interval_lengths must contain at least one value.")
    if any(int(value) <= 0 for value in interval_lengths):
        raise ValueError("interval_lengths must contain only positive values.")

    mask = active_mask.clone()
    current = int((~mask).sum().item())

    while current != target:
        make_zero = current < target
        remaining = abs(target - current)

        # ZERO correction draws from ACTIVE regions; ACTIVE correction draws
        # from ZERO regions.
        source_value = True if make_zero else False
        source_intervals = _mask_intervals(mask, value=source_value)

        if not source_intervals:
            raise RuntimeError(
                "Control-mask correction has no source interval to replace."
            )

        bank_index = int(torch.randint(
            0,
            len(interval_lengths),
            (1,),
            generator=rng,
            device=mask.device,
        ).item())
        requested_length = min(
            int(interval_lengths[bank_index]),
            remaining,
        )

        # Prefer source intervals that can accommodate the complete requested
        # replacement.  If none can, use the longest available source interval.
        eligible = [
            interval
            for interval in source_intervals
            if (interval[1] - interval[0]) >= requested_length
        ]

        if eligible:
            interval_index = int(torch.randint(
                0,
                len(eligible),
                (1,),
                generator=rng,
                device=mask.device,
            ).item())
            source_start, source_end = eligible[interval_index]
            replacement_length = requested_length
        else:
            maximum_source_length = max(
                end - start
                for start, end in source_intervals
            )
            longest = [
                interval
                for interval in source_intervals
                if (interval[1] - interval[0]) == maximum_source_length
            ]
            interval_index = int(torch.randint(
                0,
                len(longest),
                (1,),
                generator=rng,
                device=mask.device,
            ).item())
            source_start, source_end = longest[interval_index]
            replacement_length = min(
                remaining,
                maximum_source_length,
            )

        source_length = source_end - source_start

        # Keep one source-state sample on both sides whenever the interval is
        # long enough. This prevents the correction from unnecessarily merging
        # the replacement with neighbouring regions.
        if source_length >= replacement_length + 2:
            minimum_start = source_start + 1
            maximum_start = source_end - replacement_length - 1
        else:
            minimum_start = source_start
            maximum_start = source_end - replacement_length

        if maximum_start > minimum_start:
            offset = int(torch.randint(
                0,
                maximum_start - minimum_start + 1,
                (1,),
                generator=rng,
                device=mask.device,
            ).item())
            replacement_start = minimum_start + offset
        else:
            replacement_start = minimum_start

        replacement_end = replacement_start + replacement_length
        mask[replacement_start:replacement_end] = not source_value

        current += replacement_length if make_zero else -replacement_length

    return mask

def _binary_intervals(mask: np.ndarray, *, value: bool) -> list[list[int]]:
    """Return contiguous [start, end) intervals having the requested value."""

    array = np.asarray(mask, dtype=bool).reshape(-1)
    intervals: list[list[int]] = []
    start: int | None = None

    for index, item in enumerate(array):
        if bool(item) == bool(value) and start is None:
            start = index
        elif bool(item) != bool(value) and start is not None:
            intervals.append([start, index])
            start = None

    if start is not None:
        intervals.append([start, int(array.size)])

    return intervals

def generate_control_batch_with_metadata(
    num_series: int,
    series_length: int,
    sampling_time: float,
    *,
    rng: torch.Generator,
    config: ControlConfig = ControlConfig(),
    input_dim: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate random-phase multi-sines with reproducible contiguous zero regions."""

    if num_series <= 0:
        raise ValueError("num_series must be positive.")
    if series_length <= 0:
        raise ValueError("series_length must be positive.")
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if not 0.0 <= config.zero_probability <= 1.0:
        raise ValueError("zero_probability must lie in [0, 1].")

    interval_lengths = _control_interval_lengths_samples(
        sampling_time=sampling_time,
        durations=tuple(config.zero_interval_durations),
    )

    time = sampling_time * torch.arange(
        series_length,
        dtype=dtype,
        device=device,
    )
    harmonics = torch.arange(
        1,
        int(config.num_harmonics) + 1,
        dtype=dtype,
        device=device,
    )
    phases = 2.0 * torch.pi * torch.rand(
        num_series,
        input_dim,
        int(config.num_harmonics),
        dtype=dtype,
        device=device,
        generator=rng,
    )

    time_grid = time[None, None, :, None]
    harmonic_grid = harmonics[None, None, None, :]
    phase_grid = phases[:, :, None, :]
    sine_terms = torch.sin(
        2.0
        * torch.pi
        * float(config.base_frequency)
        * harmonic_grid
        * time_grid
        + phase_grid
    )
    controls = sine_terms.sum(dim=-1).transpose(1, 2).contiguous()

    active_masks = torch.ones(
        (num_series, series_length, input_dim),
        dtype=torch.bool,
        device=device,
    )
    target_zero_count = int(round(float(config.zero_probability) * series_length))

    for series_index in range(num_series):
        for channel_index in range(input_dim):
            proposal = torch.ones(series_length, dtype=torch.bool, device=device)

            cursor = 0

            if config.max_contiguous_interval_duration is None:
                max_interval = max(interval_lengths)
            else:
                max_interval = max(
                    1,
                    int(round(
                        config.max_contiguous_interval_duration /
                        sampling_time
                    ))
                )

            while cursor < series_length:

                # ------------------------------------------------------------------
                # Decide whether the NEXT interval is ACTIVE or ZERO.
                # ------------------------------------------------------------------

                zero_region = bool(
                    torch.rand((), generator=rng, device=device).item()
                    < float(config.zero_probability)
                )

                bank_index = int(
                    torch.randint(
                        0,
                        len(interval_lengths),
                        (1,),
                        generator=rng,
                        device=device,
                    ).item()
                )

                interval_length = int(interval_lengths[bank_index])

                # ------------------------------------------------------------------
                # Prevent pathological merging of identical neighbouring intervals.
                # ------------------------------------------------------------------

                if cursor > 0:

                    previous_state = bool(proposal[cursor - 1].item())

                    proposed_state = not zero_region

                    if previous_state == proposed_state:

                        # Determine current contiguous interval length.
                        contiguous = 1
                        idx = cursor - 2

                        while idx >= 0 and bool(proposal[idx].item()) == previous_state:
                            contiguous += 1
                            idx -= 1

                        # If merging would exceed the configured maximum interval,
                        # force a state toggle.
                        if contiguous + interval_length > max_interval:
                            zero_region = not zero_region

                end = min(cursor + interval_length, series_length)

                if zero_region:
                    proposal[cursor:end] = False

                cursor = end

            active_masks[series_index, :, channel_index] = _correct_control_zero_count(
                proposal,
                target_zero_count=target_zero_count,
                interval_lengths=interval_lengths,
                rng=rng,
            )

    controls = controls * active_masks.to(dtype=controls.dtype)

    if config.scale_input:
        rms = torch.sqrt(torch.mean(controls**2, dim=1, keepdim=True))
        if torch.any(rms <= 1e-12):
            raise ValueError(
                "Cannot RMS-normalize a control trajectory/channel containing only zeros. "
                "Use zero_probability < 1 when scale_input=True."
            )
        controls = controls * (float(config.amplitude) / rms)

    masks_cpu = active_masks.detach().cpu().numpy()
    zero_intervals: list[list[list[list[int]]]] = []
    active_intervals: list[list[list[list[int]]]] = []
    actual_zero_fraction: list[list[float]] = []

    for series_index in range(num_series):
        zero_series: list[list[list[int]]] = []
        active_series: list[list[list[int]]] = []
        fraction_series: list[float] = []
        for channel_index in range(input_dim):
            mask = masks_cpu[series_index, :, channel_index]
            zero_series.append(_binary_intervals(mask, value=False))
            active_series.append(_binary_intervals(mask, value=True))
            fraction_series.append(float(np.mean(~mask)))
        zero_intervals.append(zero_series)
        active_intervals.append(active_series)
        actual_zero_fraction.append(fraction_series)

    metadata = {
        "requested_zero_fraction": float(config.zero_probability),
        "target_zero_count_per_series_channel": target_zero_count,
        "actual_zero_fraction_per_series_channel": actual_zero_fraction,
        "zero_interval_durations_seconds": [float(v) for v in config.zero_interval_durations],
        "zero_interval_lengths_samples": [int(v) for v in interval_lengths],
        "zero_intervals_by_series_channel": zero_intervals,
        "active_intervals_by_series_channel": active_intervals,
        "rms_scaled_after_zero_mask": bool(config.scale_input),
        "target_rms": float(config.amplitude) if config.scale_input else None,
    }
    return controls, metadata

def generate_control_batch(
    num_series: int,
    series_length: int,
    sampling_time: float,
    *,
    rng: torch.Generator,
    config: ControlConfig = ControlConfig(),
    input_dim: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate controls while preserving the legacy tensor-only public return type."""

    controls, _ = generate_control_batch_with_metadata(
        num_series=num_series,
        series_length=series_length,
        sampling_time=sampling_time,
        rng=rng,
        config=config,
        input_dim=input_dim,
        device=device,
        dtype=dtype,
    )
    return controls

def simulate_time_series_dataset(
    system: PhysicalSystem,
    simulation_config: SimulationConfig,
    initial_condition_config: InitialConditionConfig,
    control_config: ControlConfig,
    noise_config: NoiseConfig,
) -> TimeSeriesDataset:
    """
    Generate a canonical TimeSeriesDataset from a physical simulator.

    Parameters
    ----------
    system
        Deterministic physical system. Its sampling time should match
        simulation_config.sampling_time.

    simulation_config
        Dataset-level simulation configuration.

    initial_condition_config
        Initial-state sampling protocol.

    control_config
        Random control-generation protocol.

    noise_config
        Optional control/output noise protocol.

    Returns
    -------
    TimeSeriesDataset
        Canonical dataset that can be passed directly to preprocess.py.
    """

    if abs(float(system.Ts) - float(simulation_config.sampling_time)) > 1e-12:
        raise ValueError(
            "system.Ts and simulation_config.sampling_time must match. "
            f"Got {system.Ts} and {simulation_config.sampling_time}."
        )

    data_rng = make_torch_generator(
        simulation_config.seed,
        device=simulation_config.device,
    )

    x0 = system.sample_initial_states(
        num_series=simulation_config.num_series,
        generator=data_rng,
    )

    phase_segment = simulation_config.control_phase_segment_length
    if phase_segment is None:
        controls_clean, control_generation_metadata = generate_control_batch_with_metadata(
            num_series=simulation_config.num_series,
            series_length=simulation_config.series_length,
            sampling_time=simulation_config.sampling_time,
            rng=data_rng, config=control_config, input_dim=system.input_dim,
            device=simulation_config.device, dtype=simulation_config.dtype,
        )
    else:
        if phase_segment <= 0 or simulation_config.series_length % phase_segment:
            raise ValueError("series_length must be divisible by control_phase_segment_length")
        n_segments = simulation_config.series_length // phase_segment
        segments, segment_metadata = generate_control_batch_with_metadata(
            num_series=simulation_config.num_series * n_segments,
            series_length=phase_segment,
            sampling_time=simulation_config.sampling_time,
            rng=data_rng, config=control_config, input_dim=system.input_dim,
            device=simulation_config.device, dtype=simulation_config.dtype,
        )
        controls_clean = segments.reshape(
            simulation_config.num_series, simulation_config.series_length, system.input_dim
        )
        control_generation_metadata = {
            "phase_segment_length": phase_segment,
            "phase_segments_per_series": n_segments,
            "segment_generation": segment_metadata,
        }

    states_clean, outputs_clean = system.simulate(
        x0=x0,
        controls=controls_clean,
        integrator=simulation_config.integrator,
        substeps=simulation_config.integrator_substeps,
    )

    states_ph_clean = None

    el_to_ph_state = getattr(system, "el_to_ph_state", None)

    if callable(el_to_ph_state):
        states_ph_clean = el_to_ph_state(states_clean)

    controls_final = controls_clean
    outputs_final = outputs_clean
    noise_metadata: dict[str, Any] = {
        "applied": False,
    }

    if noise_config is not None:
        controls_final, outputs_final, applied_noise_metadata = apply_noise_config(
            controls=controls_clean,
            outputs=outputs_clean,
            noise_config=noise_config,
            rng=data_rng,
        )
        noise_metadata = {
            "applied": True,
            **applied_noise_metadata,
        }

    dataset = build_time_series_dataset(
        controls=controls_final,
        outputs=outputs_final,
        states=states_clean,
        states_ph=states_ph_clean,
        sampling_time=simulation_config.sampling_time,
        metadata={
            "source": simulation_config.source_name,
            "data_seed": simulation_config.seed,
            "simulation_config": _serializable_dataclass(simulation_config),
            "initial_condition_config": _serializable_dataclass(initial_condition_config),
            "control_config": _serializable_dataclass(control_config),
            "control_generation": control_generation_metadata,
            "system": system.system_metadata(),
            "noise": noise_metadata,
            "clean_data_available_in_generation": True,
            **simulation_config.metadata,
        },
    )

    return dataset

def build_time_series_dataset(
    *,
    controls: torch.Tensor,
    outputs: torch.Tensor,
    states: torch.Tensor | None,
    states_ph: torch.Tensor | None = None,
    sampling_time: float,
    metadata: dict[str, Any] | None = None,
) -> TimeSeriesDataset:
    """
    Assemble tensors into the canonical TimeSeriesDataset dataclass.

    Parameters
    ----------
    controls
        Control trajectories [I, T, input_dim].

    outputs
        Output trajectories [I, T, output_dim].

    states
        Optional state trajectories [I, T, state_dim].

    states_ph
        Optional canonical port-Hamiltonian state trajectories
        [I, T, ph_state_dim].

    sampling_time
        Sampling period Ts.

    metadata
        Dataset metadata.

    Returns
    -------
    TimeSeriesDataset
        Canonical dataset consumed by the rest of the framework.
    """

    controls_np = _to_numpy_3d(controls, name="controls")
    outputs_np = _to_numpy_3d(outputs, name="outputs")

    if controls_np.shape[:2] != outputs_np.shape[:2]:
        raise ValueError(
            "controls and outputs must have matching [I, T] dimensions. "
            f"Got {controls_np.shape[:2]} and {outputs_np.shape[:2]}."
        )

    states_np = None
    if states is not None:
        states_np = _to_numpy_3d(states, name="states")
        if states_np.shape[:2] != controls_np.shape[:2]:
            raise ValueError(
                "states and controls must have matching [I, T] dimensions. "
                f"Got {states_np.shape[:2]} and {controls_np.shape[:2]}."
            )
    states_ph_np = None
    if states_ph is not None:
        states_ph_np = _to_numpy_3d(
            states_ph,
            name="states_ph",
        )
        if states_ph_np.shape[:2] != controls_np.shape[:2]:
            raise ValueError(
                "states_ph and controls must have matching [I, T] dimensions. "
                f"Got {states_ph_np.shape[:2]} and {controls_np.shape[:2]}."
            )
        
    num_series = controls_np.shape[0]
    series_length = controls_np.shape[1]

    time_vector = float(sampling_time) * np.arange(series_length, dtype=float)
    time = np.tile(time_vector[None, :], (num_series, 1))
    sample_index = np.tile(
        np.arange(series_length, dtype=int)[None, :],
        (num_series, 1),
    )
    series_ids = np.arange(num_series, dtype=int)

    return TimeSeriesDataset(
        controls=controls_np,
        outputs=outputs_np,
        states=states_np,
        states_ph=states_ph_np,
        time=time,
        sample_index=sample_index,
        Ts=float(sampling_time),
        series_ids=series_ids,
        metadata={} if metadata is None else dict(metadata),
    )

def _to_numpy_3d(
    tensor: torch.Tensor,
    *,
    name: str,
) -> np.ndarray:
    """Convert a tensor-like object to a NumPy array with shape [I, T, C]."""

    array = torch.as_tensor(tensor).detach().cpu().numpy()

    if array.ndim == 2:
        array = array[..., None]

    if array.ndim != 3:
        raise ValueError(f"Expected {name} with shape [I, T, C], got {array.shape}.")

    return array

def _serializable_dataclass(instance: Any) -> dict[str, Any]:
    """Return a dictionary representation for dataclass-like configuration."""

    if hasattr(instance, "__dataclass_fields__"):
        result = asdict(instance)
    else:
        result = dict(instance)

    for key, value in list(result.items()):
        if isinstance(value, torch.dtype):
            result[key] = str(value).replace("torch.", "")

    return result

def preprocess_time_series(
    dataset: TimeSeriesDataset,
    alignment: str = "causal",
    source_alignment: str = "simultaneous",
) -> TimeSeriesDataset:
    """
    Convert a source trajectory to the requested sample alignment.

    Parameters
    ----------
    dataset
        Canonical time-series dataset.

    alignment
        Required output alignment.

        "causal"
            Each row contains (u_k, y_{k+1}, x_{k+1}).

        "simultaneous"
            Each row contains (u_k, y_k, x_k).

    source_alignment
        Alignment already present in the supplied dataset.

        Simulated datasets produced by PhysicalSystem.simulate() use
        "simultaneous".
    """

    valid_alignments = {
        "simultaneous",
        "causal",
    }

    if alignment not in valid_alignments:
        raise ValueError(
            f"Unknown requested alignment '{alignment}'. "
            f"Expected one of {sorted(valid_alignments)}."
        )

    if source_alignment not in valid_alignments:
        raise ValueError(
            f"Unknown source alignment '{source_alignment}'. "
            f"Expected one of {sorted(valid_alignments)}."
        )

    dataset = deepcopy(dataset)

    if source_alignment == alignment:
        dataset.metadata["source_sample_alignment"] = source_alignment
        dataset.metadata["sample_alignment"] = alignment
        dataset.metadata["sample_definition"] = (
            "(u_k, y_{k+1}, x_{k+1})"
            if alignment == "causal"
            else "(u_k, y_k, x_k)"
        )
        return dataset

    if source_alignment == "simultaneous" and alignment == "causal":
        dataset.controls = dataset.controls[:, :-1, :]
        dataset.outputs = dataset.outputs[:, 1:, :]

        if dataset.states is not None:
            dataset.states = dataset.states[:, 1:, :]

        if dataset.states_ph is not None:
            dataset.states_ph = dataset.states_ph[:, 1:, :]

        dataset.time = dataset.time[:, 1:]
        dataset.sample_index = dataset.sample_index[:, 1:]

        dataset.metadata["source_sample_alignment"] = "simultaneous"
        dataset.metadata["sample_alignment"] = "causal"
        dataset.metadata["sample_definition"] = (
            "(u_k, y_{k+1}, x_{k+1})"
        )

        return dataset

    raise ValueError(
        "Conversion from causal source data to simultaneous data is not "
        "supported because the original y_0 and x_0 samples are absent."
    )



def split_msd(dataset: TimeSeriesDataset, train: int, validation: int, test: int, seed: int) -> SplitDataset:
    """Exact permutation-and-slice protocol from data_generation.py."""
    if train + validation + test != dataset.num_series:
        raise ValueError("Split counts must exhaust the generated trajectories")
    permutation = np.random.default_rng(seed).permutation(dataset.num_series)
    def select(indices):
        """Retrieve select for the requested experiment. Used when preparing MSD or CT experiment datasets."""
        return TimeSeriesDataset(
            controls=dataset.controls[indices], outputs=dataset.outputs[indices],
            states=None if dataset.states is None else dataset.states[indices],
            states_ph=None if dataset.states_ph is None else dataset.states_ph[indices],
            time=dataset.time[indices], sample_index=dataset.sample_index[indices],
            Ts=dataset.Ts, series_ids=dataset.series_ids[indices],
            metadata=dict(dataset.metadata))
    return SplitDataset(select(permutation[:train]),
                        select(permutation[train:train+validation]),
                        select(permutation[train+validation:]))


def apply_msd_measurement_noise(clean: SplitDataset, config: ExperimentConfig) -> SplitDataset:
    """Independent output noise on each split, with the original torch RNG protocol."""
    from oeph_random import apply_noise_config
    def noisy(ds, snr, seed):
        """Handle noisy for the experiment workflow. Used when preparing MSD or CT experiment datasets."""
        if snr is None:
            return deepcopy(ds)
        rng = make_torch_generator(seed, device=config.data.simulation_device)
        u, y, metadata = apply_noise_config(
            torch.as_tensor(ds.controls), torch.as_tensor(ds.outputs),
            NoiseConfig(output_snr_db=snr, add_output_noise=True, add_control_noise=False), rng=rng)
        out = deepcopy(ds)
        out.controls = u.detach().cpu().numpy()
        out.outputs = y.detach().cpu().numpy()
        out.metadata = {**out.metadata, "measurement_noise": metadata}
        return out
    d = config.data
    return SplitDataset(noisy(clean.train, d.noise_snr_train, d.noise_seed_train),
                        noisy(clean.validation, d.noise_snr_validation, d.noise_seed_validation),
                        noisy(clean.test, d.noise_snr_test, d.noise_seed_test))


def generate_msd_data(config: ExperimentConfig) -> tuple[SplitDataset, dict]:
    """Generate clean MSD trajectories, then split and independently corrupt outputs."""
    d = config.data
    dtype = getattr(torch, d.simulation_dtype)
    physical = MSD2Config(**{**d.physical_parameters,
        "init_range_pos": d.initial_position_range,
        "init_range_vel": d.initial_velocity_range})
    system = MSD2(physical, sampling_time=d.sampling_time, device=d.simulation_device, dtype=dtype)
    sim = SimulationConfig(num_series=d.train_trajectories+d.validation_trajectories+d.test_trajectories,
        series_length=d.trajectory_length, sampling_time=d.sampling_time,
        integrator=d.simulation_integrator, integrator_substeps=d.simulation_substeps,
        seed=d.data_seed, device=d.simulation_device, dtype=dtype)
    control = ControlConfig(base_frequency=d.control_base_frequency,
        num_harmonics=d.control_num_harmonics, zero_probability=d.control_zero_probability,
        scale_input=d.control_scale_input, amplitude=d.control_amplitude,
        zero_interval_durations=d.control_zero_interval_durations)
    if d.generation_mode == "continuous":
        return _generate_continuous_msd(config, system, sim, control)
    if d.generation_mode != "independent":
        raise ValueError(f"Unknown MSD generation mode: {d.generation_mode}")
    complete = simulate_time_series_dataset(system, sim,
        InitialConditionConfig(position_range=d.initial_position_range, velocity_range=d.initial_velocity_range),
        control, NoiseConfig(add_output_noise=False, add_control_noise=False))
    split = split_msd(complete, d.train_trajectories, d.validation_trajectories, d.test_trajectories, d.split_seed)
    # Preserve first-N trajectory selection, rather than resampling the dataset.
    if d.active_train_trajectories > d.train_trajectories:
        raise ValueError("active_train_trajectories exceeds available training trajectories")
    def select_train(ds):
        """Retrieve train for the requested experiment. Used when preparing MSD or CT experiment datasets."""
        n=d.active_train_trajectories
        return TimeSeriesDataset(ds.controls[:n],ds.outputs[:n],
            None if ds.states is None else ds.states[:n],
            None if ds.states_ph is None else ds.states_ph[:n],
            ds.time[:n],ds.sample_index[:n],ds.Ts,ds.series_ids[:n],dict(ds.metadata))
    split.train=select_train(split.train)
    target_alignment='simultaneous' if d.output_before_integration else 'causal'
    split=SplitDataset(*(preprocess_time_series(getattr(split,k),target_alignment,d.source_alignment)
                         for k in ('train','validation','test')))
    split=apply_msd_measurement_noise(split,config)
    return split, {"physical_parameters":asdict(physical), "data_seed":d.data_seed,
                   "split_seed":d.split_seed,"sample_alignment":target_alignment}


def prepare_ct_data(config: ExperimentConfig) -> tuple[SplitDataset, dict]:
    """Reference CT CSV protocol: train record; validation prefix; full test record."""
    d=config.data
    candidates=sorted(Path(d.data_root).rglob(d.dataset_filename))
    if len(candidates)!=1:
        raise ValueError(f"Expected exactly one {d.dataset_filename} under {d.data_root}, found {len(candidates)}")
    path=candidates[0]
    # The original parser uses module constants; parameterize its exact logic here.
    records, recorded_ts = _read_ct_records(path,d.train_columns,d.validation_columns,d.sampling_time_column)
    if config.training.square_outputs:
        records=[r.copy() for r in records]
        for record in records:
            record[:,1]=record[:,1]**2
    ncfg=d.normalization
    normalize=ncfg.enabled
    mean=records[0].mean(axis=0) if normalize else np.zeros(2)
    std=records[0].std(axis=0,ddof=0)
    # Notebook/reference protocol: std + 1e-15; alternative: floor at epsilon.
    scale=(std+1e-15 if ncfg.statistics_source=='reference' else np.maximum(std,ncfg.epsilon)) if normalize else np.ones(2)
    if not ncfg.normalize_input: mean[0],scale[0]=0.,1.
    if not ncfg.normalize_output: mean[1],scale[1]=0.,1.
    ts=d.sampling_time if config.model.model_sampling_time is None else config.model.model_sampling_time
    if len(records[1]) < d.validation_raw_samples:
        raise ValueError("Validation record is shorter than validation_raw_samples")
    metadata={"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_path":str(path),"recorded_sampling_time":recorded_ts,
              "model_sampling_time":ts,"resampled":False,"normalize":normalize,
              "square_outputs":config.training.square_outputs,"mean":mean,"scale":scale,
              "validation_raw_samples":d.validation_raw_samples,"validation_test_overlap":True,
              "raw_lengths":[len(r) for r in records],
              "evaluation_inverse":"sqrt" if config.training.square_outputs else "affine",
              "reference_protocol":ncfg.statistics_source=='reference',
              "normalization_rule":"population_std_plus_1e-15" if ncfg.statistics_source=='reference' else "population_std_floor"}
    def dataset(raw,series_id):
        """Handle dataset for the experiment workflow. Used when preparing MSD or CT experiment datasets."""
        values=((raw-mean)/scale).astype(np.dtype(config.model.dtype))
        n=len(raw)
        ds=TimeSeriesDataset(controls=values[None,:,:1],outputs=values[None,:,1:],
            states=None,states_ph=None,time=(np.arange(n)*ts)[None,:],
            sample_index=np.arange(n)[None,:],Ts=ts,series_ids=np.array([series_id]),
            metadata=dict(metadata,sample_alignment=d.source_alignment))
        return preprocess_time_series(ds,
            alignment='simultaneous' if d.output_before_integration else 'causal',
            source_alignment=d.source_alignment)
    split=SplitDataset(dataset(records[0],0),dataset(records[1][:d.validation_raw_samples],1),dataset(records[1],1))
    metadata['aligned_lengths']={name:getattr(split,name).series_length for name in ('train','validation','test')}
    metadata['sample_alignment']='simultaneous' if d.output_before_integration else 'causal'
    metadata['output_before_integration']=d.output_before_integration
    metadata['first_test_target_raw_index']=config.windows.history_length+(1 if not d.output_before_integration and d.source_alignment=='simultaneous' else 0)
    metadata['test_target_samples']=split.test.series_length-config.windows.history_length
    return split,metadata


def _read_ct_records(path,train_columns,validation_columns,sampling_time_column):
    """Load ct records for experiment processing. Used when preparing MSD or CT experiment datasets."""
    content=Path(path).read_text(encoding='utf-8-sig')
    dialect=csv.Sniffer().sniff(content[:8192],delimiters=',;\t')
    reader=csv.DictReader(io.StringIO(content),dialect=dialect)
    reader.fieldnames=[str(s).strip().lstrip('\ufeff') for s in reader.fieldnames]
    required=set(train_columns+validation_columns+(sampling_time_column,))
    if not required.issubset(reader.fieldnames):
        raise ValueError(f"Missing columns: {required-set(reader.fieldnames)}")
    rows=list(reader)
    periods=[float(r[sampling_time_column]) for r in rows if r.get(sampling_time_column,'').strip()]
    if not periods or not all(math.isfinite(t) and t>0 for t in periods):
        raise ValueError('CSV Ts must be finite and positive')
    if not np.allclose(periods,periods[0],rtol=0,atol=1e-12):
        raise ValueError('Nonuniform CSV Ts')
    records=[]
    for uc,yc in (train_columns,validation_columns):
        values=[]; ended=False
        for index,row in enumerate(rows):
            u,y=row.get(uc,'').strip(),row.get(yc,'').strip()
            if bool(u)!=bool(y):raise ValueError(f'Unpaired missing value in {uc}/{yc}, row {index}')
            if not u:ended=True;continue
            if ended:raise ValueError(f'Interior missing sample in {uc}/{yc}, row {index}')
            pair=(float(u),float(y))
            if not all(math.isfinite(v) for v in pair):raise ValueError(f'Nonfinite measurement at row {index}')
            values.append(pair)
        if not values:raise ValueError(f'Empty record {uc}/{yc}')
        records.append(np.asarray(values,dtype=np.float64))
    return records,periods[0]


def prepare_data(config: ExperimentConfig) -> tuple[SplitDataset,dict]:
    """Single entry point for MSD or CT."""
    if config.data.system=='MSD':return generate_msd_data(config)
    if config.data.system=='CT':return prepare_ct_data(config)
    raise ValueError(f'Unknown system {config.data.system}')


def _generate_continuous_msd(config, system, sim, control):
    """Exact continuous MSD2_16 reference split: 20+8 segments, independent test."""
    from dataclasses import replace
    d=config.data
    segment=d.control_phase_segment_length
    train_length=d.continuous_train_segments*segment
    validation_length=d.continuous_validation_segments*segment
    test_length=d.continuous_test_segments*segment
    clean_noise=NoiseConfig(add_output_noise=False,add_control_noise=False)
    init=InitialConditionConfig(position_range=d.initial_position_range,
                                velocity_range=d.initial_velocity_range)
    trainval=simulate_time_series_dataset(system,replace(sim,num_series=1,
        series_length=train_length+validation_length,seed=d.data_seed,
        control_phase_segment_length=segment),init,control,clean_noise)
    test=simulate_time_series_dataset(system,replace(sim,num_series=1,
        series_length=test_length,seed=d.data_seed+1,
        control_phase_segment_length=segment),init,control,clean_noise)
    def take(source,start,end,series_id,label):
        """Handle take for the experiment workflow. Used when preparing MSD or CT experiment datasets."""
        return TimeSeriesDataset(source.controls[:,start:end,:].copy(),
            source.outputs[:,start:end,:].copy(),
            None if source.states is None else source.states[:,start:end,:].copy(),
            None if source.states_ph is None else source.states_ph[:,start:end,:].copy(),
            source.time[:,start:end].copy(),source.sample_index[:,start:end].copy(),
            source.Ts,np.array([series_id]),
            {**source.metadata,"dataset_generation_mode":"continuous","benchmark_split":label})
    split=SplitDataset(take(trainval,0,train_length,0,'train'),
        take(trainval,train_length,train_length+validation_length,1,'validation'),
        take(test,0,test_length,2,'test'))
    alignment='simultaneous' if d.output_before_integration else 'causal'
    split=SplitDataset(*(preprocess_time_series(getattr(split,k),alignment,d.source_alignment)
                         for k in ('train','validation','test')))
    split=apply_msd_measurement_noise(split,config)
    return split,{"physical_parameters":asdict(system.config),"data_seed":d.data_seed,
                  "split_seed":None,"sample_alignment":alignment,
                  "generation_mode":"continuous"}


def inverse_output(values, metadata, *, physical_units=True):
    """Undo CT affine normalization and optional squared-output transform."""
    values = np.asarray(values)
    if not physical_units:
        return values
    mean = np.asarray(metadata['mean'])[1]
    scale = np.asarray(metadata['scale'])[1]
    unscaled = values * scale + mean
    if metadata.get('square_outputs', False):
        return np.sqrt(np.maximum(unscaled, 0.0))
    return unscaled


def inverse_input(values, metadata):
    """Undo affine input normalization."""
    return np.asarray(values) * np.asarray(metadata['scale'])[0] + np.asarray(metadata['mean'])[0]


def transform_for_method(raw_split, raw_metadata, config):
    """Build an independent model-coordinate view from cached raw time series.

    The cached arrays are never modified. CT follows the original transform
    order exactly: square outputs, calculate train-record statistics, normalize.
    MSD's original data path performs neither operation; retain that behavior
    unless explicitly requested through method preprocessing.
    """
    from copy import deepcopy
    split = deepcopy(raw_split)
    metadata = deepcopy(raw_metadata)
    ncfg = config.data.normalization
    squared = bool(config.training.square_outputs)
    if config.data.system == 'MSD' and not (squared or ncfg.enabled):
        return split, metadata
    if squared:
        for name in ('train', 'validation', 'test'):
            ds = getattr(split, name)
            ds.outputs = ds.outputs ** 2
    if ncfg.enabled:
        # CT uses the first complete training record, matching prepare_ct_data.
        train = split.train
        raw = np.concatenate((train.controls[0], train.outputs[0]), axis=-1)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0, ddof=0)
        scale = (std + 1e-15 if ncfg.statistics_source == 'reference'
                 else np.maximum(std, ncfg.epsilon))
        if not ncfg.normalize_input:
            mean[0], scale[0] = 0., 1.
        if not ncfg.normalize_output:
            mean[1], scale[1] = 0., 1.
    else:
        mean, scale = np.zeros(2), np.ones(2)
    for name in ('train', 'validation', 'test'):
        ds = getattr(split, name)
        if ncfg.enabled:
            ds.controls = ((ds.controls - mean[0]) / scale[0]).astype(np.dtype(config.model.dtype))
            ds.outputs = ((ds.outputs - mean[1]) / scale[1]).astype(np.dtype(config.model.dtype))
        if config.data.system == "CT" and not ncfg.enabled:
            ds.controls = ds.controls.astype(np.dtype(config.model.dtype))
            ds.outputs = ds.outputs.astype(np.dtype(config.model.dtype))
        ds.metadata.update(normalize=ncfg.enabled, square_outputs=squared,
                           mean=mean.copy(), scale=scale.copy())
    metadata.update(normalize=ncfg.enabled, square_outputs=squared,
                    mean=mean, scale=scale,
                    evaluation_inverse='sqrt' if squared else 'affine',
                    reference_protocol=ncfg.statistics_source == 'reference',
                    normalization_rule=('population_std_plus_1e-15' if ncfg.statistics_source == 'reference'
                                        else 'population_std_floor'))
    return split, metadata
