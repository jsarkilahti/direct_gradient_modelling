"""Compute prediction and rollout evaluation metrics for saved experiment results."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch


@dataclass
class Evaluation:
    predictions: list
    targets: list
    mse: float
    rmse: float
    nrmse: float
    trajectory_metrics: list
    segment_metrics: list
    trajectory_status: list
    failure_reasons: list
    valid_prediction_samples: list
    completed: int
    rejected: int
    truncated: int
    sampling_time: float
    skip_initial_samples: int
    total_rollout_time: float


def _samples(duration, ts, name, allow_zero):
    """Handle samples for the experiment workflow. Used to evaluate saved predictions."""
    ratio = float(duration) / ts
    n = int(round(ratio))
    if not np.isclose(ratio, n, rtol=0, atol=1e-9):
        raise ValueError(f'{name}={duration} is not an integer multiple of Ts={ts}')
    if n < 0 or (not allow_zero and n == 0):
        raise ValueError(f'{name} produces an invalid sample count')
    return n


def _metrics(p, y, denominator=None):
    """Handle metrics for the experiment workflow. Used to evaluate saved predictions."""
    mse = float(np.mean((p - y) ** 2))
    rmse = float(np.sqrt(mse))
    scale = float(np.std(y if denominator is None else denominator))
    return mse, rmse, rmse / scale if scale > 0 else float('nan')


def _valid_prefix(model, hu, hy, fu, initial_state):
    """Handle valid prefix for the experiment workflow. Used to evaluate saved predictions."""
    requires_grad = bool(getattr(model, 'requires_forward_grad', False))
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    predictions = []
    failure = None
    with context:
        try:
            if getattr(model, 'initial_state_source', 'state_encoder') == 'state_encoder':
                x = model.encode(hu, hy)
            else:
                if initial_state is None:
                    raise ValueError('true_ph_state requires initial state')
                x = initial_state.to(device=fu.device, dtype=fu.dtype)
            if not bool(torch.isfinite(x).all()):
                raise FloatingPointError('non-finite initial PH state entering test rollout')
            for k in range(fu.shape[1]):
                try:
                    output_first = bool(getattr(model, 'output_before_integration', False))
                    if output_first:
                        y = model.output(x)
                    else:
                        x = model.step(x, fu[:, k, :])
                        if not bool(torch.isfinite(x).all()):
                            raise FloatingPointError(f'non-finite PH state after test step {k+1}')
                        y = model.output(x)
                    if not bool(torch.isfinite(y).all()):
                        raise FloatingPointError(f'non-finite predicted output at test step {k+1}')
                    predictions.append(y.detach().cpu())
                    if output_first and k + 1 < fu.shape[1]:
                        x = model.step(x, fu[:, k, :])
                        if not bool(torch.isfinite(x).all()):
                            raise FloatingPointError(f'non-finite PH state after test step {k+1}')
                    if requires_grad:
                        x = x.detach()
                except FloatingPointError as exc:
                    failure = str(exc)
                    break
        except FloatingPointError as exc:
            failure = str(exc)
    if predictions:
        return torch.stack(predictions, dim=1).squeeze(0).numpy(), failure
    return np.empty((0, int(getattr(model, 'dim_y'))), dtype=np.float32), failure


def evaluate(model, data, *, history_length, skip_initial_time=0.0,
             segment_length_time=None, denominator='evaluated_suffix',
             output_transform=None):
    """Evaluate all trajectories; full_record matches CT notebook normalization."""
    if denominator not in ('full_record', 'evaluated_suffix'):
        raise ValueError('Unknown denominator')
    ts = float(data.Ts)
    if not np.isfinite(ts) or ts <= 0:
        raise ValueError('Dataset must have positive finite Ts')
    skip = _samples(skip_initial_time, ts, 'skip_initial_time', True)
    segment = None if segment_length_time is None else _samples(segment_length_time, ts, 'segment_length_time', False)
    model.eval()
    device = next(model.parameters()).device
    predictions, targets, traj, segments, statuses, failures, counts = [], [], [], [], [], [], []
    global_p, global_y = [], []
    started = perf_counter()
    for i in range(data.num_series):
        controls = np.asarray(data.controls[i])
        outputs = np.asarray(data.outputs[i])
        if len(controls) <= history_length:
            raise ValueError('Trajectory shorter than history')
        hu = torch.as_tensor(controls[None, :history_length], dtype=torch.float32, device=device)
        hy = torch.as_tensor(outputs[None, :history_length], dtype=torch.float32, device=device)
        fu = torch.as_tensor(controls[None, history_length:], dtype=torch.float32, device=device)
        initial = None
        if getattr(model, 'initial_state_source', 'state_encoder') == 'true_ph_state':
            ix = history_length if getattr(model, 'output_before_integration', False) else history_length - 1
            initial = torch.as_tensor(data.states_ph[i, ix][None], dtype=torch.float32, device=device)
        valid, reason = _valid_prefix(model, hu, hy, fu, initial)
        count = len(valid)
        counts.append(count)
        failures.append(reason)
        target = outputs[history_length:]
        padded = np.full(target.shape, np.nan, dtype=np.float32)
        padded[:count] = valid
        predictions.append(padded)
        targets.append(target)
        if count <= skip:
            statuses.append('rejected')
            continue
        statuses.append('complete' if count == len(target) else 'truncated')
        p, y = padded[skip:count], target[skip:count]
        if output_transform is not None:
            p, y = output_transform(p), output_transform(y)
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            statuses[-1] = 'rejected'
            failures[-1] = 'Nonfinite evaluation inverse'
            continue
        full = data.outputs[i] if denominator == 'full_record' else y
        if denominator == 'full_record' and output_transform is not None:
            full = output_transform(full)
        mse, rmse, nrmse = _metrics(p, y, full)
        traj.append(dict(index=i, mse=mse, rmse=rmse, nrmse=nrmse))
        global_p.append(p.reshape(-1))
        global_y.append(y.reshape(-1))
        if segment is not None:
            for start in range(0, len(y), segment):
                stop = min(start + segment, len(y))
                smse, srmse, snrmse = _metrics(p[start:stop], y[start:stop])
                segments.append(dict(trajectory=i, start=start, stop=stop,
                                     mse=smse, rmse=srmse, nrmse=snrmse))
    if not global_p:
        raise FloatingPointError('No finite evaluable test trajectory')
    p = np.concatenate(global_p)
    y = np.concatenate(global_y)
    if denominator == 'full_record':
        full = data.outputs
        if output_transform is not None:
            full = output_transform(full)
    else:
        full = y
    mse, rmse, nrmse = _metrics(p, y, full)
    return Evaluation(predictions, targets, mse, rmse, nrmse, traj, segments,
                      statuses, failures, counts, statuses.count('complete'),
                      statuses.count('rejected'), statuses.count('truncated'),
                      ts, skip, perf_counter() - started)
