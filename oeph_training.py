"""Execute training, validation, checkpoint selection, and study-level experiment orchestration."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from oeph_definitions import TrainingConfig, OptimizerConfig, WindowConstructionMetadata, FitResult, TestResult
from oeph_windowing import _construct_window_dataset

def _make_data_loader(
    data: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None,
    require_true_ph_state: bool,
    output_before_integration: bool = False,
) -> DataLoader:
    """Construct one deterministic tensor DataLoader."""

    expected_alignment = "simultaneous" if output_before_integration else "causal"
    actual_alignment = getattr(getattr(data, "metadata", None), "sample_alignment", None)
    if actual_alignment is not None and actual_alignment != expected_alignment:
        raise ValueError(f"Rollout expects {expected_alignment} windows; got {actual_alignment}")

    tensors = [
        _float_tensor(data.history_controls),
        _float_tensor(data.history_outputs),
        _float_tensor(data.future_controls),
        _float_tensor(data.future_outputs),
    ]

    if require_true_ph_state:
        history_states_ph = getattr(data, "history_states_ph", None)
        if history_states_ph is None:
            raise ValueError(
                "true_ph_state initialisation requires data.history_states_ph."
            )
        if history_states_ph.shape[1] != data.history_controls.shape[1]:
            raise ValueError(
                "history_states_ph and history_controls must have equal history length."
            )
        # Output-first: use the first future state x[k].
        # Integrate-first: use the last history state, then advance to the target.
        if output_before_integration:
            future_states_ph = getattr(data, "future_states_ph", None)
            if future_states_ph is None:
                raise ValueError("Output-first true-state initialization requires future_states_ph")
            tensors.append(_float_tensor(future_states_ph[:, 0, :]))
        else:
            tensors.append(_float_tensor(history_states_ph[:, -1, :]))

    dataset = TensorDataset(*tensors)

    generator: torch.Generator | None = None

    if shuffle:
        if seed is None:
            raise ValueError(
                "A seed must be provided when DataLoader shuffling is enabled."
            )

        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _float_tensor(array: np.ndarray) -> torch.Tensor:
    """Handle float tensor for the experiment workflow. Used during experiment training, validation, or result collection."""
    return torch.as_tensor(array, dtype=torch.float32)


def _require_finite_scalar(value: torch.Tensor, name: str) -> None:
    """Raise immediately when a scalar training quantity is non-finite."""
    if not bool(torch.isfinite(value.detach()).item()):
        raise FloatingPointError(f"non-finite {name}: {float(value.detach())}")


def _clip_and_require_finite_gradients(
    model: torch.nn.Module,
    gradient_clip_norm: float | None,
) -> tuple[float, bool]:
    """Clip finite gradients and reject NaN/Inf gradients before optimiser.step()."""
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    raw_gradient_norm = torch.linalg.vector_norm(
    torch.stack([
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
        for parameter in parameters
    ]),
    ord=2,
)

    #print(f"gradient norm before clipping: {float(raw_gradient_norm):.6e}")
    clipped = False

    if gradient_clip_norm is not None:
        clipped = float(raw_gradient_norm) > gradient_clip_norm

        try:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=gradient_clip_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            raise FloatingPointError(
                "non-finite gradient norm before optimiser step"
            ) from exc

        #print(f"gradient clip: {'yes' if clipped else 'no'}")
    else:
        for parameter in parameters:
            if not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(
                    "non-finite gradient before optimiser step"
                )

        #print("gradient clip: no")

    post_gradient_norm = torch.linalg.vector_norm(
        torch.stack([
            torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
            for parameter in parameters
        ]),
        ord=2,
    )

    #print(f"gradient norm after clipping: {float(post_gradient_norm):.6e}")
    return float(raw_gradient_norm), clipped


def _require_finite_parameters(model: torch.nn.Module) -> None:
    """Reject an optimiser update that creates NaN/Inf parameters."""
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not bool(torch.isfinite(parameter).all().item()):
            raise FloatingPointError(
                f"non-finite parameter after optimiser step: {name}"
            )


def _sync_device(device):
    """Synchronize asynchronous accelerators before timing boundaries."""
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def _time_start(device):
    """Handle time start for the experiment workflow. Used during experiment training, validation, or result collection."""
    _sync_device(device)
    return perf_counter()


def _time_stop(device, start):
    """Handle time stop for the experiment workflow. Used during experiment training, validation, or result collection."""
    _sync_device(device)
    return perf_counter() - start


def _timing_summary(epoch_records):
    """Original benchmark convention: steady state excludes the first epoch."""
    result = {'timing_epoch_records': epoch_records,
              'timing_measured_epochs': len(epoch_records),
              'timing_steady_state_epochs': max(0, len(epoch_records) - 1)}
    for name in ('forward', 'backward', 'optimizer', 'validation', 'train', 'total'):
        values = [r[name + '_seconds'] for r in epoch_records]
        result['timing_' + name + '_total_seconds'] = float(sum(values))
        result['timing_' + name + '_steady_mean_seconds'] = (
            float(np.mean(values[1:])) if len(values) > 1 else float('nan'))
    return result


def _train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
    timing: dict | None = None,
) -> float:
    """Perform one minibatch training epoch and collect its timing measurements."""
    model.train()

    loss_sum = 0.0
    sample_count = 0

    gradient_norms: list[float] = []
    gradient_clipped: list[bool] = []

    for batch in data_loader:
        history_controls, history_outputs, future_controls, targets = (
            tensor.to(device) for tensor in batch[:4]
        )
        initial_state = batch[4].to(device) if len(batch) == 5 else None

        optimiser.zero_grad(set_to_none=True)
        t_forward = _time_start(device) if timing is not None else None
        predictions = model(
            history_controls,
            history_outputs,
            future_controls,
            initial_state=initial_state,
        )
        loss = torch.mean((predictions - targets) ** 2)
        if timing is not None:
            timing["forward_seconds"] += _time_stop(device, t_forward)
        _require_finite_scalar(loss, "training loss")

        t_backward = _time_start(device) if timing is not None else None
        loss.backward()
        if timing is not None:
            timing["backward_seconds"] += _time_stop(device, t_backward)
        raw_gradient_norm, clipped = _clip_and_require_finite_gradients(
            model,
            gradient_clip_norm,
        )

        gradient_norms.append(raw_gradient_norm)
        gradient_clipped.append(clipped)

        t_optimizer = _time_start(device) if timing is not None else None
        optimiser.step()
        if timing is not None:
            timing["optimizer_seconds"] += _time_stop(device, t_optimizer)
        _require_finite_parameters(model)

        batch_size = targets.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("train_data contains no windows.")

    gradient_norm_array = np.asarray(gradient_norms, dtype=float)

    return loss_sum / sample_count


def _validation_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the model on validation windows for checkpoint selection."""
    model.eval()

    loss_sum = 0.0
    sample_count = 0
    requires_forward_grad = bool(
        getattr(model, "requires_forward_grad", False)
    )

    context = torch.enable_grad() if requires_forward_grad else torch.no_grad()

    with context:
        for batch in data_loader:
            history_controls, history_outputs, future_controls, targets = (
                tensor.to(device) for tensor in batch[:4]
            )
            initial_state = batch[4].to(device) if len(batch) == 5 else None
            predictions = model(
                history_controls,
                history_outputs,
                future_controls,
                initial_state=initial_state,
            )
            loss = torch.mean((predictions - targets) ** 2)
            _require_finite_scalar(loss, "validation loss")

            batch_size = targets.shape[0]
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("validation_data contains no windows.")

    return loss_sum / sample_count



def _step_scheduler(scheduler: Any, validation_loss: float) -> None:
    """Handle step scheduler for the experiment workflow. Used during experiment training, validation, or result collection."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)
    else:
        scheduler.step()

def make_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build one optimizer using the exact configured PyTorch hyperparameters."""
    name = config.name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                betas=config.betas, eps=config.eps,
                                weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.name}")

def make_scheduler(optimizer: torch.optim.Optimizer, config: OptimizerConfig):
    """Construct scheduler from the supplied configuration. Used during experiment training, validation, or result collection."""
    if config.scheduler_type is None:
        return None
    if config.scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.scheduler_parameters)
    if config.scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler_parameters)
    raise ValueError(f"Unsupported scheduler: {config.scheduler_type}")

def _loader(data, config, *, shuffle, seed, model):
    """Load loader for experiment processing. Used during experiment training, validation, or result collection."""
    return _make_data_loader(data, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=config.num_workers, seed=seed,
        require_true_ph_state=(getattr(model, "initial_state_source", "state_encoder") == "true_ph_state"),
        output_before_integration=getattr(model, "output_before_integration", False))

def _phase_seeds(master_seed: int, count: int) -> list[int]:
    """Exact distinct-phase-seed generator from the original CT notebook."""
    rng = np.random.default_rng(master_seed)
    seeds, used = [], set()
    while len(seeds) < count:
        seed = int(rng.integers(0, 2**32))
        if seed not in used:
            seeds.append(seed)
            used.add(seed)
    return seeds


def _train_phased_ct(model, train_data, validation_data, config, optimizer,
                     scheduler, validation_full_data, validation_metadata=None, progress=None):
    """CT notebook continuation: phase RNG, checkpoint and optimizer restoration."""
    from torch.utils.data import Subset
    device = next(model.parameters()).device
    phases = tuple(config.fit_phases)
    if sum(phases) != config.epochs or any(n <= 0 for n in phases):
        raise ValueError("CT fit_phases must be positive and sum to epochs")
    if config.prediction_horizon_schedule or config.state_derivative_limit_schedule:
        raise ValueError("Original CT phase training does not use curriculum or derivative clipping")
    base_loader = _loader(train_data, config, shuffle=False, seed=None, model=model)
    dataset = base_loader.dataset
    validation_loader = _loader(validation_data, config, shuffle=False, seed=None, model=model)
    full_loader = None
    if config.validation_mode != 'windowed':
        if validation_full_data is None:
            raise ValueError("Full validation data required")
        n = validation_data.metadata.history_length
        if config.validation_mode == 'mixed':
            full_windows = _construct_window_dataset(validation_full_data,
                [np.array([0], dtype=int) for _ in range(validation_full_data.num_series)],
                WindowConstructionMetadata(history_length=n,
                    prediction_length=validation_full_data.series_length - n,
                    sample_alignment=validation_data.metadata.sample_alignment))
            full_loader = _loader(full_windows, config, shuffle=False, seed=None, model=model)

    validation_records = []
    last_components = {}
    criterion = ('full_rollout_nrmse' if config.validation_mode == 'full_rollout'
                 else 'window_mse' if config.validation_mode == 'windowed'
                 else 'weighted_window_full_mse')
    def evaluate():
        """Evaluate a trained model against the requested dataset split."""
        nonlocal last_components
        last_components = {}
        try:
            if config.validation_mode == 'full_rollout':
                from oeph_evaluation import evaluate as evaluate_trajectory
                from oeph_data import inverse_output
                denominator = config.validation_nrmse_denominator
                transform = (lambda x: inverse_output(x, validation_metadata)) if config.validation_physical_metrics else None
                if transform is not None and validation_metadata is None:
                    raise ValueError('Physical validation requires normalization metadata')
                result = evaluate_trajectory(model, validation_full_data,
                    history_length=validation_data.metadata.history_length,
                    skip_initial_time=config.validation_skip_initial_time,
                    segment_length_time=None, denominator=denominator,
                    output_transform=transform)
                value = result.nrmse if result.completed == validation_full_data.num_series else float('nan')
                last_components = dict(full_rollout_nrms=result.nrmse,
                    completed=result.completed, expected=validation_full_data.num_series)
                return value
            window_loss = _validation_epoch(model, validation_loader, device)
            last_components = dict(window_mse=window_loss)
            if config.validation_mode == 'windowed':
                return window_loss
            if config.validation_mode == 'mixed':
                full_loss = _validation_epoch(model, full_loader, device)
                w = config.validation_full_rollout_weight
                value = (1-w)*window_loss + w*full_loss
                last_components.update(full_rollout_mse=full_loss, combined_mse=value)
                return value
            raise ValueError(f'Unknown validation mode: {config.validation_mode}')
        except FloatingPointError:
            return float('nan')

    histories = {'train': [], 'validation': [], 'lr': []}
    best, best_value, best_epoch = None, float('inf'), 0
    updates, executed_epochs = 0, 0
    ledger = []
    epoch_timings = []
    seeds = (_phase_seeds(config.phase_shuffle_master_seed, len(phases))
             if config.phase_shuffle_master_seed is not None
             else [config.shuffle_seed] * len(phases))
    lr_schedule = config.learning_rate_schedule
    if lr_schedule is not None and 0 not in lr_schedule:
        raise ValueError('learning_rate_schedule must contain epoch 0')

    def snapshot(epoch):
        """Handle snapshot for the experiment workflow. Used during experiment training, validation, or result collection."""
        return dict(model=deepcopy(model.state_dict()),
                    optimizer=deepcopy(optimizer.state_dict()),
                    scheduler=deepcopy(scheduler.state_dict()) if scheduler else None,
                    histories=deepcopy(histories), epoch=epoch, updates=updates)

    rng = None
    ids = None
    for phase_index, (budget, seed) in enumerate(zip(phases, seeds)):
        print(f"[PHASE] {phase_index+1}/{len(phases)} epochs={budget} shuffle_seed={seed}", flush=True)
        if rng is None or config.reshuffle_at_phase_boundary:
            rng = np.random.default_rng(seed)
            ids = np.arange(len(dataset))
        if config.validate_initial:
            value = evaluate()
            validation_records.append(dict(phase=phase_index+1, phase_epoch=0,
                epoch=executed_epochs, value=value, criterion=criterion,
                **deepcopy(last_components)))
            if np.isfinite(value) and value <= best_value:
                best_value, best_epoch = value, executed_epochs
                best = snapshot(best_epoch)
                if progress is not None:
                    progress(best['model'], best_epoch, best_value)
                print(f"[BEST] initial validation={best_value:.8g}; checkpoint saved", flush=True)
        for phase_epoch in range(1, budget + 1):
            epoch_start = _time_start(device)
            epoch_parts = dict(forward_seconds=0.0, backward_seconds=0.0, optimizer_seconds=0.0)
            if lr_schedule is not None:
                rate = lr_schedule[max(k for k in lr_schedule if k <= executed_epochs)]
                for group in optimizer.param_groups:
                    group['lr'] = float(rate)
            epoch_lr = float(optimizer.param_groups[0]['lr'])
            if config.shuffle:
                rng.shuffle(ids)
            losses = []
            value = float('nan')
            model.train()
            for start in range(0, len(ids), config.batch_size):
                selected = ids[start:start + config.batch_size]
                if len(selected) < config.batch_size and config.drop_last:
                    break
                batch = [dataset[int(i)] for i in selected]
                tensors = [torch.stack([item[j] for item in batch]).to(device)
                           for j in range(len(batch[0]))]
                hu, hy, u, y = tensors[:4]
                initial_state = tensors[4] if len(tensors) > 4 else None
                optimizer.zero_grad()
                t_forward = _time_start(device)
                prediction = model(hu, hy, u, initial_state=initial_state)
                loss = torch.stack([torch.nn.functional.mse_loss(y[:, t], prediction[:, t])
                                    for t in range(y.shape[1])]).mean()
                epoch_parts["forward_seconds"] += _time_stop(device, t_forward)
                t_backward = _time_start(device)
                loss.backward()
                epoch_parts["backward_seconds"] += _time_stop(device, t_backward)
                if config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                t_optimizer = _time_start(device)
                optimizer.step()
                epoch_parts["optimizer_seconds"] += _time_stop(device, t_optimizer)
                updates += 1
                value = float(loss.detach())
                if not np.isfinite(value):
                    break
                losses.append(value)
            if not np.isfinite(value):
                break
            t_validation = _time_start(device)
            val = evaluate()
            epoch_parts["validation_seconds"] = _time_stop(device, t_validation)
            epoch_parts["train_seconds"] = sum(epoch_parts[k] for k in ("forward_seconds", "backward_seconds", "optimizer_seconds"))
            epoch_parts["total_seconds"] = _time_stop(device, epoch_start)
            epoch_timings.append(dict(epoch=executed_epochs+1, **epoch_parts))
            validation_records.append(dict(phase=phase_index+1, phase_epoch=phase_epoch,
                epoch=executed_epochs+1, value=val, criterion=criterion,
                **deepcopy(last_components)))
            train_loss = float(np.mean(losses)) if losses else float('nan')
            histories['train'].append(train_loss)
            histories['validation'].append(val)
            histories['lr'].append(float(optimizer.param_groups[0]['lr']))
            if scheduler is not None:
                _step_scheduler(scheduler, val)
            improved = np.isfinite(val) and val <= best_value
            if improved:
                best_value, best_epoch = val, executed_epochs+1
                best = snapshot(best_epoch)
                if progress is not None:
                    progress(best['model'], best_epoch, best_value)
            print(f"[EPOCH] {executed_epochs+1:4d}/{config.epochs} "
                  f"phase={phase_index+1}/{len(phases)} lr={epoch_lr:.6g} "
                  f"patience={config.early_stopping_patience} "
                  f"train={train_loss:.8g} best_train={min([row['training_mse'] for row in ledger] + [train_loss]):.8g} "
                  f"val={val:.8g} best_val={best_value:.8g} "
                  f"{'BEST SAVED' if improved else ''}", flush=True)
            ledger.append(dict(phase=phase_index+1, phase_epoch=phase_epoch,
                shuffle_seed=seed, executed_epoch=executed_epochs,
                learning_rate=epoch_lr, updates=updates,
                training_mse=train_loss, validation_value=val,
                validation_criterion=criterion, validation_components=deepcopy(last_components),
                improved=bool(improved)))
            executed_epochs += 1
        if best is None:
            raise RuntimeError('No finite initial/epoch validation checkpoint')
        if phase_index + 1 == len(phases) or config.reload_best_at_phase_boundary:
            model.load_state_dict(best['model'])
            optimizer.load_state_dict(best['optimizer'])
            if scheduler is not None and best['scheduler'] is not None:
                scheduler.load_state_dict(best['scheduler'])
            if phase_index + 1 < len(phases) and config.restore_history_between_phases:
                histories = deepcopy(best['histories'])
                updates = best['updates']
    model.state_derivative_limit = float('inf')
    # Histories in a checkpoint may be truncated by phase-boundary restoration.
    # The chronological ledger is the authoritative record of executed epochs.
    chronological_train = [row['training_mse'] for row in ledger]
    chronological_validation = [row['validation_value'] for row in ledger]
    chronological_lr = [row['learning_rate'] for row in ledger]
    result = FitResult(model, deepcopy(best['model']), best_value, best_epoch,
        chronological_train, chronological_validation, chronological_lr, executed_epochs)
    result.phase_ledger = ledger
    result.timing_metrics = _timing_summary(epoch_timings)
    result.reload_best_at_phase_boundary = config.reload_best_at_phase_boundary
    result.reshuffle_at_phase_boundary = config.reshuffle_at_phase_boundary
    result.phase_shuffle_seeds = seeds
    result.phase_shuffle_master_seed = config.phase_shuffle_master_seed
    result.shuffle_seed_policy = ('distinct_phase_seeds' if config.phase_shuffle_master_seed is not None else 'legacy_repeat')
    result.validation_records = validation_records
    result.validation_criterion = criterion
    result.executed_epochs = executed_epochs
    result.optimizer_updates = updates
    result.continuation_checkpoint = dict(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), scheduler=deepcopy(scheduler.state_dict()) if scheduler is not None else None, best_model=deepcopy(best["model"]), best_epoch=best_epoch, best_validation=best_value, histories=deepcopy(histories), executed_epochs=executed_epochs, optimizer_updates=updates)
    return result


class ExperimentRejectedError(RuntimeError):
    """Reference-compatible rejection when no finite checkpoint is selectable."""


def train(model: torch.nn.Module, train_data, validation_data, config: TrainingConfig,
          optimizer: torch.optim.Optimizer, scheduler=None, validation_full_data=None,
          validation_metadata=None, progress=None) -> FitResult:
    """Train with the reference MSE, batching, LR scheduling and best-checkpoint rule."""
    if config.shuffle_seed_policy != "fixed":
        raise ValueError("Only the reference fixed shuffle-seed policy is supported")
    if config.fit_phases is not None and getattr(config, "system", None) == "CT":
        return _train_phased_ct(model, train_data, validation_data, config,
                                optimizer, scheduler, validation_full_data, validation_metadata, progress)

    device = torch.device(next(model.parameters()).device)
    model.to(device)
    schedule = getattr(config, "prediction_horizon_schedule", None)
    if schedule is not None:
        if not isinstance(train_data, Mapping) or not isinstance(validation_data, Mapping):
            raise ValueError("A horizon schedule requires horizon-indexed window datasets")
        if 0 not in schedule:
            raise ValueError("Horizon schedule must start at epoch offset zero")
        final_horizon = schedule[max(schedule)]
        loaders = {h: (_loader(train_data[h], config, shuffle=config.shuffle, seed=config.shuffle_seed, model=model),
                       _loader(validation_data[h], config, shuffle=False, seed=None, model=model))
                   for h in sorted(set(schedule.values()))}
    else:
        loaders = {None: (_loader(train_data, config, shuffle=config.shuffle, seed=config.shuffle_seed, model=model),
                          _loader(validation_data, config, shuffle=False, seed=None, model=model))}
        final_horizon = None
    full_loader = None
    if config.validation_mode != "windowed":
        if validation_full_data is None:
            raise ValueError("Full-rollout validation requires validation_full_data")
        reference = next(iter(validation_data.values())) if isinstance(validation_data, Mapping) else validation_data
        n = reference.metadata.history_length
        horizon = validation_full_data.series_length - n
        if horizon <= 0:
            raise ValueError("Validation trajectory is shorter than encoder history")
        full_windows = _construct_window_dataset(validation_full_data,
            [np.array([0], dtype=int) for _ in range(validation_full_data.num_series)],
            WindowConstructionMetadata(history_length=n, prediction_length=horizon,
                                       sample_alignment=reference.metadata.sample_alignment))
        full_loader = _make_data_loader(full_windows, batch_size=1, shuffle=False,
            num_workers=0, seed=None,
            require_true_ph_state=(getattr(model, "initial_state_source", "state_encoder") == "true_ph_state"),
            output_before_integration=getattr(model, "output_before_integration", False))
    phase_lengths = config.fit_phases or (config.epochs,)
    if sum(phase_lengths) != config.epochs or any(n <= 0 for n in phase_lengths):
        raise ValueError("fit_phases must contain positive lengths summing to epochs")
    lr_schedule = getattr(config, "learning_rate_schedule", None)
    if lr_schedule is not None and 0 not in lr_schedule:
        raise ValueError("learning_rate_schedule must contain epoch 0")
    derivative_schedule = getattr(config, "state_derivative_limit_schedule", None)
    train_losses, val_losses, lrs = [], [], []
    best_loss, best_epoch = float("inf"), 0
    best_state = deepcopy(model.state_dict())
    patience_count = 0
    training_terminated_nonfinite = False
    phase_end_epochs = set(np.cumsum(phase_lengths)[:-1].tolist())
    phase_best_value = float('inf')
    phase_best_model = None
    phase_best_optimizer = None
    phase_best_scheduler = None
    epoch_timings = []
    for epoch in range(1, config.epochs + 1):
        epoch_start = _time_start(device)
        epoch_parts = dict(forward_seconds=0.0, backward_seconds=0.0, optimizer_seconds=0.0)
        elapsed = epoch - 1
        if lr_schedule is not None:
            lr = lr_schedule[max(k for k in lr_schedule if k <= elapsed)]
            for group in optimizer.param_groups:
                group["lr"] = float(lr)
        horizon = schedule[max(k for k in schedule if k <= elapsed)] if schedule is not None else None
        final_active = horizon == final_horizon
        train_loader, val_loader = loaders[horizon]
        if derivative_schedule is not None:
            model.state_derivative_limit = derivative_schedule[max(k for k in derivative_schedule if k <= elapsed)]
        else:
            model.state_derivative_limit = float("inf")
        try:
            t_train = _time_start(device)
            loss = _train_epoch(model, train_loader, optimizer, device, config.gradient_clip_norm, timing=epoch_parts)
            epoch_parts["train_seconds"] = _time_stop(device, t_train)
            t_validation = _time_start(device)
            window_loss = _validation_epoch(model, val_loader, device)
            if full_loader is None:
                val_loss = window_loss
            else:
                full_loss = _validation_epoch(model, full_loader, device)
                if config.validation_mode == "full_rollout":
                    val_loss = full_loss
                elif config.validation_mode == "mixed":
                    w = config.validation_full_rollout_weight
                    val_loss = (1.0 - w) * window_loss + w * full_loss
                else:
                    raise ValueError(f"Unknown validation mode: {config.validation_mode}")
            epoch_parts["validation_seconds"] = _time_stop(device, t_validation)
            epoch_parts["total_seconds"] = _time_stop(device, epoch_start)
            epoch_timings.append(dict(epoch=epoch, **epoch_parts))
        except FloatingPointError as exc:
            training_terminated_nonfinite = True
            print(f"Non-finite MSD training/validation at epoch {epoch}: {exc}")
            break
        if final_active and lr_schedule is None:
            _step_scheduler(scheduler, val_loss)
        train_losses.append(loss)
        val_losses.append(val_loss)
        lrs.append(float(optimizer.param_groups[0]["lr"]))
        if np.isfinite(val_loss) and val_loss < phase_best_value:
            phase_best_value = val_loss
            phase_best_model = deepcopy(model.state_dict())
            phase_best_optimizer = deepcopy(optimizer.state_dict())
            phase_best_scheduler = deepcopy(scheduler.state_dict()) if scheduler is not None else None
        if final_active:
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = deepcopy(model.state_dict())
                patience_count = 0
                if progress is not None:
                    progress(best_state, best_epoch, best_loss)
            else:
                patience_count += 1
        else:
            patience_count = 0
        print(f"[EPOCH] {epoch:4d}/{config.epochs} horizon={horizon if horizon is not None else 'fixed'} "
              f"lr={lrs[-1]:.6g} patience={patience_count}/{config.early_stopping_patience} "
              f"train={loss:.8g} best_train={min(train_losses):.8g} "
              f"val={val_loss:.8g} best_val={best_loss:.8g} "
              f"{'BEST SAVED' if best_epoch == epoch else ''}", flush=True)
        if epoch in phase_end_epochs:
            if config.reload_best_at_phase_boundary and phase_best_model is not None:
                model.load_state_dict(phase_best_model)
                optimizer.load_state_dict(phase_best_optimizer)
                if scheduler is not None and phase_best_scheduler is not None:
                    scheduler.load_state_dict(phase_best_scheduler)
            if config.reshuffle_at_phase_boundary:
                for train_loader_for_horizon, _ in loaders.values():
                    sampler = getattr(train_loader_for_horizon, 'sampler', None)
                    generator = getattr(sampler, 'generator', None)
                    if generator is not None:
                        generator.manual_seed(int(config.shuffle_seed))
            phase_best_value = float('inf')
            phase_best_model = None
            phase_best_optimizer = None
            phase_best_scheduler = None
        if final_active and config.early_stopping_patience is not None and patience_count >= config.early_stopping_patience:
            break
    if best_epoch == 0:
        if training_terminated_nonfinite:
            raise ExperimentRejectedError("Non-finite MSD training before a finite final-horizon checkpoint")
        raise RuntimeError("No finite validation checkpoint at final prediction horizon")
    model.load_state_dict(best_state)
    model.state_derivative_limit = float("inf")
    result = FitResult(model, best_state, best_loss, best_epoch, train_losses, val_losses, lrs, len(train_losses))
    result.timing_metrics = _timing_summary(epoch_timings)
    result.reload_best_at_phase_boundary = config.reload_best_at_phase_boundary
    result.reshuffle_at_phase_boundary = config.reshuffle_at_phase_boundary
    result.training_terminated_nonfinite = training_terminated_nonfinite
    result.continuation_checkpoint = dict(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), scheduler=deepcopy(scheduler.state_dict()) if scheduler is not None else None, best_model=deepcopy(best_state), best_epoch=best_epoch, best_validation=best_loss, histories=dict(train=deepcopy(train_losses), validation=deepcopy(val_losses), lr=deepcopy(lrs)))
    return result

def validate(model: torch.nn.Module, validation_data, config: TrainingConfig) -> float:
    """Reference window-averaged validation MSE."""
    loader = _loader(validation_data, config, shuffle=False, seed=None, model=model)
    return _validation_epoch(model, loader, next(model.parameters()).device)

def test(model: torch.nn.Module, test_data, *, history_length: int,
         skip_initial_samples: int = 0, output_scale: float = 1.0,
         skip_initial_time: float | None = None,
         segment_length_time: float | None = None,
         denominator: str = 'evaluated_suffix',
         output_transform=None) -> TestResult:
    """Original guarded, trajectory-wise rollout with optional time-based scoring."""
    from oeph_evaluation import evaluate
    if skip_initial_time is None:
        skip_initial_time = skip_initial_samples * float(test_data.Ts)
    if output_scale != 1.0 and output_transform is not None:
        raise ValueError('Select output_scale or output_transform, not both')
    transform = output_transform or ((lambda x: x * output_scale) if output_scale != 1.0 else None)
    result = evaluate(model, test_data, history_length=history_length,
        skip_initial_time=skip_initial_time,
        segment_length_time=segment_length_time,
        denominator=denominator, output_transform=transform)
    out = TestResult(result.predictions, result.targets, result.mse, result.rmse, result.nrmse)
    out.trajectory_metrics = result.trajectory_metrics
    out.segment_metrics = result.segment_metrics
    out.trajectory_status = result.trajectory_status
    out.failure_reasons = result.failure_reasons
    out.valid_prediction_samples = result.valid_prediction_samples
    out.completed = result.completed
    out.rejected = result.rejected
    out.truncated = result.truncated
    out.skip_initial_samples = result.skip_initial_samples
    return out




def _checkpoint_window_metrics(model, windows, config, *, inverse=None, denominator_targets=None):
    """Evaluate every selected window at the restored best checkpoint, without shuffling.

    MSE pools all target samples (not the mean of per-batch RMSEs). NRMSE
    uses the same population-standard-deviation convention as oeph_stats.
    """
    from oeph_stats import regression_metrics
    loader = _loader(windows, config, shuffle=False, seed=None, model=model)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    predictions, targets = [], []
    context = torch.enable_grad() if getattr(model, 'requires_forward_grad', False) else torch.no_grad()
    try:
        with context:
            for batch in loader:
                hu, hy, fu, target = (a.to(device) for a in batch[:4])
                initial = batch[4].to(device) if len(batch) > 4 else None
                prediction = model(hu, hy, fu, initial_state=initial)
                predictions.append(prediction.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
    finally:
        model.train(was_training)
    if inverse is not None:
        predictions = [inverse(a) for a in predictions]
        targets = [inverse(a) for a in targets]
    return regression_metrics(predictions, targets, denominator_targets=denominator_targets)


def _checkpoint_rollout_metrics(model, series, *, history_length, skip_initial_time,
                                inverse=None, denominator_mode='evaluated'):
    """Evaluate complete trajectories using the existing guarded rollout kernel."""
    from oeph_stats import regression_metrics
    evaluation = test(model, series, history_length=history_length,
                      skip_initial_time=skip_initial_time)
    predictions, targets = [], []
    for i, status in enumerate(evaluation.trajectory_status):
        if status not in ('complete', 'truncated'):
            continue
        stop = evaluation.valid_prediction_samples[i]
        start = evaluation.skip_initial_samples
        if stop > start:
            predictions.append(evaluation.predictions[i][start:stop])
            targets.append(evaluation.targets[i][start:stop])
    if not predictions:
        raise FloatingPointError('No valid rollout samples for checkpoint metrics')
    if inverse is not None:
        predictions = [inverse(a) for a in predictions]
        targets = [inverse(a) for a in targets]
    denominator = None
    if denominator_mode == 'full_record':
        denominator = series.outputs if inverse is None else inverse(series.outputs)
    return regression_metrics(predictions, targets, denominator_targets=denominator)


# Stage 5: direct study execution.
"""Direct Stage-5 study execution; no study-definition or rule hierarchy."""
from dataclasses import replace
import numpy as np
from oeph_definitions import ExperimentConfig, RunResult, StudyResult, validate_configuration
from oeph_data import prepare_data
from oeph_windowing import construct_window_split_dataset
from oeph_system import build_model


_COMPONENT_NAMES = ('encoder', 'interconnection', 'dissipation', 'port', 'hamiltonian')


def _component_seed_mapping(study, base_seed):
    """Resolve reproducible component seeds, shared across Hamiltonian methods.

    Keys use the component names accepted by oeph_system.build_model.
    """
    mode = study.seed_mode
    if mode == 'same_seed_all_components':
        return {name: int(base_seed) for name in _COMPONENT_NAMES}
    if mode == 'independent_component_seeds':
        return {name: 5 * int(base_seed) + i for i, name in enumerate(_COMPONENT_NAMES)}
    if mode == 'manual_seeding':
        seeds = study.manual_component_seeds
        if set(seeds) != set(_COMPONENT_NAMES):
            raise ValueError('MANUAL_COMPONENT_SEEDS must contain exactly: ' + ', '.join(_COMPONENT_NAMES))
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0 for v in seeds.values()):
            raise ValueError('Manual component seeds must be nonnegative integers')
        return {name: int(seeds[name]) for name in _COMPONENT_NAMES}
    raise ValueError(f'Unsupported SEED_MODE: {mode!r}')


def _seed_list(study):
    """Handle seed list for the experiment workflow. Used during experiment training, validation, or result collection."""
    if study.explicit_model_seeds is not None:
        seeds = tuple(int(x) for x in study.explicit_model_seeds)
    else:
        seeds = tuple(int(x) for x in np.random.default_rng(study.seed_master).integers(0, 2**31 - 1, size=study.num_seeds))
    if not seeds or any(x < 0 for x in seeds):
        raise ValueError('Model seeds must be nonempty nonnegative integers')
    return seeds


def _shuffle_list(study):
    """Handle shuffle list for the experiment workflow. Used during experiment training, validation, or result collection."""
    if study.explicit_shuffle_seeds is not None:
        seeds = tuple(int(x) for x in study.explicit_shuffle_seeds)
    else:
        seeds = tuple(int(x) for x in np.random.default_rng(study.shuffle_seed_master).integers(0, 2**31 - 1, size=study.num_shuffles))
    if not seeds or any(x < 0 for x in seeds):
        raise ValueError('Shuffle seeds must be nonempty nonnegative integers')
    return seeds


def _variants(config):
    """Handle variants for the experiment workflow. Used during experiment training, validation, or result collection."""
    study = config.study
    kind = study.study_type
    if kind == 'baseline':
        return [(None, config)]
    if kind == 'prediction_horizon':
        variants = []
        for h in study.horizon_levels:
            h = int(h)
            training = config.training
            if training.prediction_horizon_schedule is not None:
                # Retain curriculum switch epochs, but cap its intermediate
                # horizons to the selected ablation horizon.
                schedule = {int(epoch): min(int(value), h)
                            for epoch, value in training.prediction_horizon_schedule.items()}
                schedule[max(schedule)] = h
                training = replace(training, prediction_horizon_schedule=schedule)
            variants.append((h, replace(config, windows=replace(config.windows, prediction_horizon=h),
                                        training=training)))
        return variants
    if kind == 'snr':
        return [(float(s), replace(config, data=replace(config.data, noise_snr_train=float(s), noise_snr_validation=float(s)))) for s in study.snr_levels]
    if kind == 'training_data_size':
        if config.data.system == 'MSD':
            return [(int(n), replace(config, data=replace(config.data, active_train_trajectories=int(n)))) for n in study.data_size_levels]
        return [(int(n), config) for n in study.data_size_levels]
    raise ValueError(f'Study execution does not implement {kind!r}; reconstruction belongs to Stage 6/7')


def _windows(split, config):
    """Handle windows for the experiment workflow. Used during experiment training, validation, or result collection."""
    w = config.windows
    return construct_window_split_dataset(split, w.history_length, w.prediction_horizon,
        stride=w.stride, train_window_fraction=w.train_fraction,
        validation_window_fraction=w.validation_fraction, test_window_fraction=w.test_fraction,
        seed=w.selection_seed)


def _limit_ct_windows(windows, count):
    """Handle limit ct windows for the experiment workflow. Used during experiment training, validation, or result collection."""
    from dataclasses import fields, replace as dc_replace
    data = windows.train
    if not 1 <= count <= data.num_windows:
        raise ValueError(f'CT training_data_size must be between 1 and {data.num_windows} windows, got {count}')
    # Deterministic prefix of the original window ordering, with no new RNG calls.
    updates = {}
    for f in fields(data):
        v = getattr(data, f.name)
        if isinstance(v, np.ndarray) and v.ndim and len(v) == data.num_windows:
            updates[f.name] = v[:count].copy()
    return dc_replace(windows, train=dc_replace(data, **updates))


def _level_folder(study_type, level):
    """Stable, filesystem-safe name for a single experiment condition."""
    import re
    name = 'baseline' if level is None else f'{study_type}_{level}'
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def execute_study(config: ExperimentConfig) -> StudyResult:
    """Execute selected study directly using the shared data/model/train/test APIs.

    Stage 5 returns in-memory results. Persistent artifacts and reconstruction
    are deliberately deferred to Stages 6 and 7.
    """
    validate_configuration(config)
    result = StudyResult(configuration=config)
    study = config.study
    # Model-seed replication is independent of the selected ablation axis.
    # Reuse the same seeds at every level for paired comparisons.
    seeds = _seed_list(study)
    # One shuffle stream preserves the system's reference shuffle seed.
    # Multiple or explicit shuffle streams are independent replications.
    shuffles = (_shuffle_list(study) if study.explicit_shuffle_seeds is not None or study.num_shuffles > 1
                else (int(config.training.shuffle_seed),))
    from oeph_analytics import parameter_counts, elapsed_time
    from oeph_stats import regression_metrics
    from oeph_reporting import save_study, reconstruct_results
    if study.study_type == "reconstruct_results":
        return reconstruct_results(study.reconstruction_directory)
    from oeph_saving import load_or_prepare_dataset, save_checkpoint
    output_directory = study.results_root / study.study_name
    variants = _variants(config)
    print(f"[STUDY] system={config.data.system} study={study.study_type} "
          f"levels={len(variants)} methods={study.methods} model_seeds={list(seeds)} "
          f"output={output_directory}", flush=True)
    for variant_index, (level, variant) in enumerate(variants):
        print(f"[LEVEL] {variant_index+1}/{len(variants)} value={level!r}", flush=True)
        # Generated MSD datasets live under the configured data root.
        # CT's data_root points to measured source files, so its cache stays in data/.
        dataset_root = (Path(variant.data.data_root)
                        if variant.data.system == 'MSD' else Path('data'))
        split, metadata, cached_path, cache_hit = load_or_prepare_dataset(
            variant, dataset_root)
        # Raw time series are shared; windows are built for each method at runtime.
        data_path = str(cached_path)
        for method_index, method in enumerate(study.methods):
            from oeph_data import transform_for_method
            from oeph_definitions import NormalizationConfig
            overrides = study.method_preprocessing.get(method, {})
            norm_overrides = {k:v for k,v in overrides.items() if k != 'square_outputs'}
            method_norm = replace(variant.data.normalization, **{
                ('enabled' if k == 'normalize' else k):v for k,v in norm_overrides.items()})
            method_variant = replace(variant,
                data=replace(variant.data, normalization=method_norm),
                training=replace(variant.training, square_outputs=overrides.get(
                    'square_outputs', variant.training.square_outputs)))
            method_split, method_metadata = transform_for_method(split, metadata, method_variant)
            if method_variant.data.system == 'CT':
                d = method_variant.data
                method_metadata['first_test_target_raw_index'] = (
                    method_variant.windows.history_length +
                    (1 if not d.output_before_integration and d.source_alignment == 'simultaneous' else 0))
                method_metadata['test_target_samples'] = (
                    method_split.test.series_length - method_variant.windows.history_length)
            if study.study_type == 'snr' and variant.data.system == 'CT':
                from copy import deepcopy
                from oeph_random import make_torch_generator, apply_noise_config
                from oeph_definitions import NoiseConfig
                for name, snr, seed in (('train', variant.data.noise_snr_train, variant.data.noise_seed_train),
                                        ('validation', variant.data.noise_snr_validation, variant.data.noise_seed_validation)):
                    ds = getattr(method_split, name)
                    rng = make_torch_generator(seed, device='cpu')
                    _, noisy_y, _ = apply_noise_config(torch.as_tensor(ds.controls), torch.as_tensor(ds.outputs),
                        NoiseConfig(output_snr_db=snr, add_output_noise=True, add_control_noise=False), rng=rng)
                    ds.outputs = noisy_y.detach().cpu().numpy()
            # All downstream operations use the method-specific model coordinates.
            print(f"[WINDOWS] Constructing in memory: history={method_variant.windows.history_length} "
                  f"horizon={method_variant.windows.prediction_horizon}", flush=True)
            windows = _windows(method_split, method_variant)
            print(f"[WINDOWS] successful: train={windows.train.num_windows} "
                  f"validation={windows.validation.num_windows} test={windows.test.num_windows}", flush=True)
            # Curriculum windows must be constructed independently for every horizon.
            # The final-horizon dataset remains the test/reference window split.
            curriculum = method_variant.training.prediction_horizon_schedule
            curriculum_windows = None
            if curriculum is not None:
                if method_variant.data.system != 'MSD':
                    raise ValueError('The reference CT trainer does not support a horizon curriculum')
                horizons = set(curriculum.values())
                if not horizons or max(horizons) != method_variant.windows.prediction_horizon:
                    raise ValueError('The final curriculum horizon must equal prediction_horizon')
                curriculum_windows = {h: _windows(method_split, replace(method_variant, windows=replace(
                    method_variant.windows, prediction_horizon=h))) for h in horizons}
            if study.study_type == 'training_data_size' and method_variant.data.system == 'CT':
                windows = _limit_ct_windows(windows, level)
            method_seeds = (int(study.selected_seeds[method]),) if method in study.selected_seeds else seeds
            for seed_index, model_seed in enumerate(method_seeds):
                for shuffle_seed in shuffles:
                    print(f"[MODEL] method={method} ({method_index+1}/{len(study.methods)}) "
                          f"seed={model_seed} ({seed_index+1}/{len(method_seeds)}) "
                          f"shuffle_seed={shuffle_seed} level={level!r}", flush=True)
                    component_seeds = _component_seed_mapping(study, model_seed)
                    print(f"[MODEL] component_seeds={component_seeds}", flush=True)
                    model = build_model(method_variant, method, model_seed, component_seeds=component_seeds)
                    optimizer_config = method_variant.training.optimizers[method]
                    optimizer = make_optimizer(model, optimizer_config)
                    scheduler = make_scheduler(optimizer, optimizer_config)
                    training_config = replace(method_variant.training, shuffle_seed=shuffle_seed)
                    training_config.learning_rate_schedule = optimizer_config.learning_rate_schedule
                    training_config.system = method_variant.data.system
                    timing = {}
                    counts = parameter_counts(model)
                    from oeph_analytics import hamiltonian_parameter_counts, benchmark_hamiltonian
                    h_counts = hamiltonian_parameter_counts(model)
                    with elapsed_time(timing, 'training_seconds'):
                        train_windows = ({h: w.train for h, w in curriculum_windows.items()}
                                         if curriculum_windows is not None else windows.train)
                        validation_windows = ({h: w.validation for h, w in curriculum_windows.items()}
                                              if curriculum_windows is not None else windows.validation)
                        checkpoint_index = f"{method}_seed{model_seed}_shuffle{shuffle_seed}"
                        def on_best(state, epoch, value):
                            """Handle on best for the experiment workflow. Used during experiment training, validation, or result collection."""
                            location = save_checkpoint(state, output_directory / _level_folder(study.study_type, level), checkpoint_index)
                            print(f"[CHECKPOINT] saved {location} epoch={epoch} "
                                  f"best_val={value:.8g}", flush=True)
                        print(f"[TRAIN] starting epochs={training_config.epochs} "
                              f"lr={optimizer.param_groups[0]['lr']:.6g} "
                              f"patience={training_config.early_stopping_patience}", flush=True)
                        fit = train(model, train_windows, validation_windows, training_config,
                            optimizer, scheduler=scheduler, validation_full_data=method_split.validation,
                            validation_metadata=method_metadata, progress=on_best)
                    with elapsed_time(timing, 'testing_seconds'):
                        evaluation = test(fit.model, method_split.test, history_length=method_variant.windows.history_length,
                            skip_initial_time=method_variant.training.test_skip_initial_time,
                            segment_length_time=method_variant.training.test_segment_length_time)
                    predictions, targets = evaluation.predictions, evaluation.targets
                    output_unit = 'Output (model coordinates)'
                    normalization = method_variant.data.normalization
                    if method_variant.data.system == 'CT':
                        from oeph_data import inverse_output
                        # A regression error is meaningful only when both arrays
                        # are expressed in the same units. The two inverse flags
                        # are independently honored, never silently overridden.
                        inverse_p = normalization.physical_unit_metrics and normalization.inverse_transform_predictions
                        inverse_y = normalization.physical_unit_metrics and normalization.inverse_transform_targets
                        if inverse_p != inverse_y:
                            raise ValueError('CT RMSE requires predictions and targets in the same units; '
                                             'set both inverse-transform switches equally')
                        if inverse_p:
                            predictions = [inverse_output(p, method_metadata) for p in predictions]
                            targets = [inverse_output(y, method_metadata) for y in targets]
                            output_unit = 'Tank level (physical units)'
                    denominator_mode = normalization.nrmse_denominator
                    if denominator_mode == 'full_record':
                        denominator_targets = method_split.test.outputs
                        if method_variant.data.system == 'CT' and normalization.physical_unit_metrics and normalization.inverse_transform_targets:
                            denominator_targets = inverse_output(denominator_targets, method_metadata)
                    elif denominator_mode in ('evaluated', 'evaluated_targets'):
                        denominator_targets = None
                    else:
                        raise ValueError(f'Unknown NRMSE denominator: {denominator_mode!r}')
                    # The reference scores valid prefixes only. Keep NaN-padded
                    # full predictions for plotting, not for numerical metrics.
                    scored_p, scored_y = [], []
                    for index, status in enumerate(evaluation.trajectory_status):
                        if status not in ('complete', 'truncated'):
                            continue
                        stop = evaluation.valid_prediction_samples[index]
                        start = evaluation.skip_initial_samples
                        if stop > start:
                            scored_p.append(predictions[index][start:stop])
                            scored_y.append(targets[index][start:stop])
                    if not scored_p:
                        raise FloatingPointError('No finite test trajectory available for scoring')
                    metrics = regression_metrics(scored_p, scored_y, denominator_targets=denominator_targets)
                    # Independent best-checkpoint metrics for all four evaluation modes.
                    # Training histories remain untouched and are still used for learning curves.
                    from oeph_data import inverse_output
                    inverse = None
                    if method_variant.data.system == 'CT' and normalization.physical_unit_metrics:
                        if normalization.inverse_transform_predictions != normalization.inverse_transform_targets:
                            raise ValueError('Prediction/target inverse transforms must match')
                        if normalization.inverse_transform_targets:
                            inverse = lambda a: inverse_output(a, method_metadata)
                    metric_denominator = normalization.nrmse_denominator
                    if metric_denominator not in ('full_record', 'evaluated', 'evaluated_targets'):
                        raise ValueError(f'Unknown NRMSE denominator: {metric_denominator}')
                    def full_reference(series):
                        """Handle full reference for the experiment workflow. Used during experiment training, validation, or result collection."""
                        if metric_denominator != 'full_record':
                            return None
                        return inverse(series.outputs) if inverse is not None else series.outputs
                    with elapsed_time(timing, 'checkpoint_metrics_seconds'):
                        with elapsed_time(timing, 'training_windowed_evaluation_seconds'):
                            train_window_metrics = _checkpoint_window_metrics(
                                fit.model, windows.train, training_config, inverse=inverse,
                                denominator_targets=full_reference(method_split.train))
                        with elapsed_time(timing, 'validation_windowed_evaluation_seconds'):
                            validation_window_metrics = _checkpoint_window_metrics(
                                fit.model, windows.validation, training_config, inverse=inverse,
                                denominator_targets=full_reference(method_split.validation))
                        with elapsed_time(timing, 'training_full_rollout_evaluation_seconds'):
                            train_rollout_metrics = _checkpoint_rollout_metrics(
                                fit.model, method_split.train, history_length=method_variant.windows.history_length,
                                skip_initial_time=0.0, inverse=inverse,
                                denominator_mode=metric_denominator)
                        with elapsed_time(timing, 'validation_full_rollout_evaluation_seconds'):
                            validation_rollout_metrics = _checkpoint_rollout_metrics(
                                fit.model, method_split.validation, history_length=method_variant.windows.history_length,
                                skip_initial_time=method_variant.training.validation_skip_initial_time,
                                inverse=inverse, denominator_mode=metric_denominator)
                    checkpoint_metrics = {
                        'training_windowed': train_window_metrics,
                        'training_full_rollout': train_rollout_metrics,
                        'validation_windowed': validation_window_metrics,
                        'validation_full_rollout': validation_rollout_metrics,
                        'testing': {k: metrics[k] for k in ('mse', 'rmse', 'nrmse', 'target_std')},
                    }
                    for split_name, values in checkpoint_metrics.items():
                        for metric_name, value in values.items():
                            metrics[f'{split_name}_{metric_name}'] = value
                    # Benchmark after evaluation; do not include the probe in training time.
                    h_timing = benchmark_hamiltonian(fit.model)
                    metrics.update(best_epoch=fit.best_validation_epoch,
                                   best_validation_loss=fit.best_validation_loss,
                                   parameter_count=counts['total'], trainable_parameter_count=counts['trainable'],
                                   hamiltonian_parameter_count=h_counts['total_parameters'],
                                   hamiltonian_trainable_parameter_count=h_counts['trainable_parameters'],
                                   **h_timing, **timing,
                                   **{k:v for k,v in fit.timing_metrics.items() if k != "timing_epoch_records"})
                    run = RunResult(method=method, model_seed=model_seed, shuffle_seed=shuffle_seed,
                        configuration={'study':study.study_type, 'level':level, 'system':method_variant.data.system,
                                       'prediction_horizon':method_variant.windows.prediction_horizon,
                                       'sampling_time':method_variant.data.sampling_time,
                                       'training_windows':windows.train.num_windows, 'output_unit':output_unit,
                                       'data_path': data_path, 'variant_index': variant_index,
                                       'seed_mode': study.seed_mode,
                                       'component_seeds': dict(component_seeds)},
                        training_loss=fit.training_loss_history,
                        validation_loss=fit.validation_loss_history,
                        metrics=metrics,
                        normalization_metadata=method_metadata)
                    # Keep full predictions, LR history and selected checkpoint accessible
                    # for numerical comparisons without forcing Stage-7 file persistence.
                    run.predictions = predictions
                    run.targets = targets
                    run.learning_rates = fit.learning_rate_history
                    run.best_model_state = fit.best_model_state
                    run.continuation_checkpoint = getattr(fit, "continuation_checkpoint", None)
                    run.training_metadata = {
                        'hamiltonian_parameter_inventory': h_counts,
                        'checkpoint_metrics': checkpoint_metrics,
                        'model_architecture': str(model),
                        'phase_ledger': getattr(fit, 'phase_ledger', None),
                        'validation_records': getattr(fit, 'validation_records', None),
                        'validation_criterion': getattr(fit, 'validation_criterion', None),
                        'phase_shuffle_seeds': getattr(fit, 'phase_shuffle_seeds', None),
                        'phase_shuffle_master_seed': getattr(fit, 'phase_shuffle_master_seed', None),
                        'shuffle_seed_policy': getattr(fit, 'shuffle_seed_policy', None),
                        'executed_epochs': getattr(fit, 'executed_epochs', None),
                        'optimizer_updates': getattr(fit, 'optimizer_updates', None),
                        'epoch_timings': fit.timing_metrics['timing_epoch_records'],
                    }
                    run.trajectory_metrics = evaluation.trajectory_metrics
                    run.segment_metrics = evaluation.segment_metrics
                    run.trajectory_status = evaluation.trajectory_status
                    result.runs.append(run)
                    print(f'{method_variant.data.system} {study.study_type}={level} {method} '
                          f'model_seed={model_seed} shuffle_seed={shuffle_seed} '
                          f'epochs={fit.epochs_completed} RMSE={metrics["rmse"]:.8g}')
    save_study(result, study.results_root / study.study_name,
               figures=study.save_figures, plot_options=study.plot_options)
    return result
