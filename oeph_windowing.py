"""Construct runtime history windows and prediction horizons from prepared time-series data."""


import numpy as np

from oeph_definitions import TimeSeriesDataset, WindowConstructionMetadata, WindowDataset, WindowSplitDataset


# =============================================================================
# Window construction metadata
# =============================================================================

# =============================================================================
# Valid window starts
# =============================================================================

def _compute_valid_window_starts(
    dataset: TimeSeriesDataset,
    history_length: int,
    prediction_length: int,
) -> tuple[list[np.ndarray], WindowConstructionMetadata]:
    """
    Compute all geometrically valid window starting indices.
    """

    if history_length <= 0:
        raise ValueError(
            "history_length must be positive."
        )

    if prediction_length <= 0:
        raise ValueError(
            "prediction_length must be positive."
        )

    T = dataset.series_length

    last_start = (
        T
        - history_length
        - prediction_length
    )

    if last_start < 0:
        raise ValueError(
            "Window is longer than the time series."
        )

    starts = np.arange(
        last_start + 1,
        dtype=int,
    )

    window_starts = [
        starts.copy()
        for _ in range(dataset.num_series)
    ]

    possible_windows = sum(
        len(indices)
        for indices in window_starts
    )

    metadata = WindowConstructionMetadata(
        history_length=history_length,
        prediction_length=prediction_length,
        possible_windows=possible_windows,
        strided_windows=possible_windows,
        selected_windows=possible_windows,
    )

    return window_starts, metadata


# =============================================================================
# Window stride
# =============================================================================

def _apply_window_stride(
    starts: list[np.ndarray],
    metadata: WindowConstructionMetadata,
    stride: int,
) -> tuple[list[np.ndarray], WindowConstructionMetadata]:
    """
    Apply a stride to valid window starting indices.
    """

    if stride <= 0:
        raise ValueError(
            "stride must be a positive integer."
        )

    strided_starts = [
        indices[::stride].copy()
        for indices in starts
    ]

    metadata.stride = stride

    metadata.strided_windows = sum(
        len(indices)
        for indices in strided_starts
    )

    metadata.selected_windows = metadata.strided_windows

    return strided_starts, metadata



def _sample_window_starts(
    starts: list[np.ndarray],
    metadata: WindowConstructionMetadata,
    window_fraction: float,
    seed: int,
) -> tuple[
    list[np.ndarray],
    WindowConstructionMetadata,
]:
    """
    Randomly sample window starting indices.
    """

    if not (0.0 < window_fraction <= 1.0):
        raise ValueError(
            "window_fraction must lie in (0,1]."
        )

    rng = np.random.default_rng(seed)

    sampled = []

    for indices in starts:

        n_keep = max(
            1,
            int(
                np.ceil(
                    window_fraction * len(indices)
                )
            ),
        )

        keep = np.sort(

            rng.choice(

                len(indices),

                size=n_keep,

                replace=False,

            )

        )

        sampled.append(
            indices[keep]
        )

    metadata.selection_fraction = window_fraction

    metadata.sampling_method = "random"

    metadata.sampling_seed = seed

    metadata.selected_windows = sum(
        len(indices)
        for indices in sampled
    )

    return sampled, metadata



# =============================================================================
# Window dataset construction
# =============================================================================

def _construct_window_dataset(
    dataset: TimeSeriesDataset,
    starts: list[np.ndarray],
    metadata: WindowConstructionMetadata,
) -> WindowDataset:
    """
    Construct the canonical WindowDataset.
    """

    history_controls = []
    history_outputs = []
    history_states = []
    history_states_ph = []

    future_controls = []
    future_outputs = []
    future_states = []
    future_states_ph = []

    history_time = []
    future_time = []

    history_sample_index = []
    future_sample_index = []

    series_ids = []
    start_indices = []

    window_ids = []

    window_id = 0

    H = metadata.history_length
    P = metadata.prediction_length

    for series in range(dataset.num_series):

        for start in starts[series]:

            history_controls.append(
                dataset.controls[
                    series,
                    start:start + H,
                ]
            )

            history_outputs.append(
                dataset.outputs[
                    series,
                    start:start + H,
                ]
            )

            if dataset.states is None:
                history_states.append(None)
            else:
                history_states.append(
                    dataset.states[
                        series,
                        start:start + H,
                    ]
                )

            if dataset.states_ph is None:
                history_states_ph.append(None)
            else:
                history_states_ph.append(
                    dataset.states_ph[
                        series,
                        start:start + H,
                    ]
                )

            future_controls.append(
                dataset.controls[
                    series,
                    start + H:start + H + P,
                ]
            )

            future_outputs.append(
                dataset.outputs[
                    series,
                    start + H:start + H + P,
                ]
            )

            if dataset.states is None:
                future_states.append(None)
            else:
                future_states.append(
                    dataset.states[
                        series,
                        start + H:start + H + P,
                    ]
                )

            if dataset.states_ph is None:
                future_states_ph.append(None)
            else:
                future_states_ph.append(
                    dataset.states_ph[
                        series,
                        start + H:start + H + P,
                    ]
                )

            history_time.append(
                dataset.time[
                    series,
                    start:start + H,
                ]
            )

            future_time.append(
                dataset.time[
                    series,
                    start + H:start + H + P,
                ]
            )

            history_sample_index.append(
                dataset.sample_index[
                    series,
                    start:start + H,
                ]
            )

            future_sample_index.append(
                dataset.sample_index[
                    series,
                    start + H:start + H + P,
                ]
            )

            series_ids.append(
                dataset.series_ids[series]
            )

            start_indices.append(start)

            window_ids.append(window_id)

            window_id += 1

    return WindowDataset(

        history_controls=np.asarray(history_controls),

        history_outputs=np.asarray(history_outputs),

        history_states=(
            None
            if dataset.states is None
            else np.asarray(history_states)
        ),

        history_states_ph=(
            None
            if dataset.states_ph is None
            else np.asarray(history_states_ph)
        ),

        future_controls=np.asarray(future_controls),

        future_outputs=np.asarray(future_outputs),

        future_states=(
            None
            if dataset.states is None
            else np.asarray(future_states)
        ),

        future_states_ph=(
            None
            if dataset.states_ph is None
            else np.asarray(future_states_ph)
        ),

        window_ids=np.asarray(window_ids),

        series_ids=np.asarray(series_ids),

        window_starts=np.asarray(start_indices),

        history_time=np.asarray(history_time),

        future_time=np.asarray(future_time),

        history_sample_index=np.asarray(history_sample_index),

        future_sample_index=np.asarray(future_sample_index),

        metadata=metadata,
    )


def construct_window_split_dataset(
    split_dataset,
    history_length: int,
    prediction_length: int,
    stride: int = 1,
    train_window_fraction: float = 1.0,
    validation_window_fraction: float = 1.0,
    test_window_fraction: float = 1.0,
    seed: int = 0,
) -> WindowSplitDataset:
    """
    Construct canonical train/validation/test WindowDatasets.
    """

    minimum_length = history_length + prediction_length

    def build(
        dataset,
        fraction: float,
        name: str,
    ) -> WindowDataset:
        """
        Construct one WindowDataset.
        """
        sample_alignment = dataset.metadata.get(
            "sample_alignment"
        )

        if sample_alignment not in {"causal", "simultaneous"}:
            raise ValueError(f"{name}: unsupported sample alignment {sample_alignment!r}")

        if dataset.series_length < minimum_length:

            raise ValueError(
                f"{name} dataset contains only "
                f"{dataset.series_length} samples per series, "
                f"but at least {minimum_length} samples are required "
                f"(history_length={history_length}, "
                f"prediction_length={prediction_length})."
            )

        #
        # Compute all geometrically valid windows
        #
        starts, metadata = _compute_valid_window_starts(
            dataset,
            history_length,
            prediction_length,
        )

        metadata.sample_alignment = sample_alignment

        #
        # Apply stride
        #
        starts, metadata = _apply_window_stride(
            starts,
            metadata,
            stride,
        )

        #
        # Randomly sample windows
        #
        starts, metadata = _sample_window_starts(
            starts,
            metadata,
            fraction,
            seed,
        )

        #
        # Construct the canonical WindowDataset
        #
        return _construct_window_dataset(
            dataset,
            starts,
            metadata,
        )

    return WindowSplitDataset(

        train=build(
            split_dataset.train,
            train_window_fraction,
            "Training",
        ),

        validation=build(
            split_dataset.validation,
            validation_window_fraction,
            "Validation",
        ),

        test=build(
            split_dataset.test,
            test_window_fraction,
            "Test",
        ),
    )