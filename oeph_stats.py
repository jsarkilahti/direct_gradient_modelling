"""Compute regression scores and across-seed statistics for experiments and plots."""
from __future__ import annotations
from collections import defaultdict
from scipy.stats import t
import numpy as np


def regression_metrics(predictions, targets, *, denominator_targets=None):
    """Compute MSE, RMSE, and NRMSE from predictions and targets."""
    p = np.concatenate([np.asarray(a).reshape(-1) for a in predictions])
    y = np.concatenate([np.asarray(a).reshape(-1) for a in targets])
    if p.shape != y.shape or not p.size:
        raise ValueError('Predictions and targets must be nonempty and have matching shapes')
    mse = float(np.mean((p-y)**2))
    rmse = float(np.sqrt(mse))
    reference = y if denominator_targets is None else np.asarray(denominator_targets).reshape(-1)
    std = float(np.std(reference, ddof=0))
    return {'mse': mse, 'rmse': rmse, 'nrmse': rmse/std if std > 0 else float('nan'),
            'target_std': std}


def restore_derived_metrics(metrics):
    """Reconstruct scalar RMSE and NRMSE from MSE and its target STD."""
    result = dict(metrics)
    for key, value in list(metrics.items()):
        if key != 'mse' and not key.endswith('_mse'):
            continue
        prefix = key[:-3]
        std = metrics.get(prefix + 'target_std')
        if std is None:
            continue
        rmse = float(np.sqrt(value))
        result[prefix + 'rmse'] = rmse
        result[prefix + 'nrmse'] = rmse / float(std) if float(std) > 0 else float('nan')
    return result


def summarize(values, confidence=0.95):
    """Summarize a sample of run-level metrics."""
    if confidence not in (0.90, 0.95, 0.99):
        raise ValueError('Reference confidence levels are 0.90, 0.95 and 0.99')
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if not n:
        return dict(n=0, best=float('nan'), median=float('nan'), q1=float('nan'), q3=float('nan'), mean=float('nan'), std=float('nan'), sem=float('nan'),
                    ci_lower=float('nan'), ci_upper=float('nan'))
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else float('nan')
    sem = std / np.sqrt(n) if n > 1 else float('nan')
    delta = t.ppf((1+confidence)/2, df=n-1)*sem if n > 1 else float('nan')
    return dict(n=n, best=float(a.min()), median=float(np.median(a)), q1=float(np.percentile(a, 25)), q3=float(np.percentile(a, 75)), mean=mean, std=std, sem=sem, ci_lower=mean-delta, ci_upper=mean+delta)


def compute_statistics(study, confidence=0.95):
    """Aggregate experiment metrics across runs for reporting and comparison."""
    groups = defaultdict(list)
    for run in study.runs:
        groups[(run.configuration.get('level'), run.method)].append(run)
    result = {}
    for (level, method), runs in groups.items():
        key = f'{method}|{level}'
        metrics = sorted(set().union(*(r.metrics.keys() for r in runs)))
        result[key] = {
            'method': method, 'level': level, 'count': len(runs),
            'metrics': {m: summarize([r.metrics.get(m, np.nan) for r in runs], confidence) for m in metrics},
            'training': _curve([r.training_loss for r in runs], confidence),
            'validation': _curve([r.validation_loss for r in runs], confidence),
        }
    return result


def _curve(histories, confidence):
    """Aggregate aligned training histories for confidence-interval plots."""
    length = max((len(h) for h in histories), default=0)
    return [summarize([h[i] if i < len(h) else np.nan for h in histories], confidence)
            for i in range(length)]


def paired_method_differences(study, confidence=0.95):
    """Paired differences at equal study level and model/shuffle repetition.

    Runs lacking a matching partner are omitted rather than paired by position.
    The sign is the first method's metric minus the second method's metric.
    """
    from itertools import combinations
    by_level = defaultdict(lambda: defaultdict(dict))
    for run in study.runs:
        level = run.configuration.get('level')
        key = (run.model_seed, run.shuffle_seed)
        if key in by_level[level][run.method]:
            raise ValueError(f'Duplicate run for level={level}, method={run.method}, seed={key}')
        by_level[level][run.method][key] = run
    results = {}
    for level, methods in by_level.items():
        for left, right in combinations(sorted(methods), 2):
            shared = sorted(set(methods[left]) & set(methods[right]))
            names = sorted(set().union(*(set(methods[left][k].metrics) & set(methods[right][k].metrics) for k in shared))) if shared else []
            metrics = {}
            for name in names:
                diffs = [methods[left][k].metrics[name] - methods[right][k].metrics[name]
                         for k in shared if name in methods[left][k].metrics and name in methods[right][k].metrics]
                metrics[name] = summarize(diffs, confidence)
            results[f'{left}-{right}|{level}'] = dict(method_a=left, method_b=right,
                level=level, paired_seeds=shared, metrics=metrics)
    return results
