"""Create paper-oriented figures from saved learning curves and prediction results."""
from pathlib import Path
import numpy as np
from oeph_plotting import paper_learning_curve_intervals, _pyplot


def plot_paper_curves(runs, destination, *, system, metric='nrmse', scale='linear',
                      train_target_std=None, validation_target_std=None,
                      confidence=0.95, ct_output_scale=None):
    """Plot paper-style learning curves from saved experiment histories."""
    curves = paper_learning_curve_intervals(
        runs, system=system, scale=scale, metric=metric,
        confidence=confidence, train_target_std=train_target_std,
        validation_target_std=validation_target_std, ct_output_scale=ct_output_scale)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for method, group in curves.items():
        for ax, field in zip(axes, ('train', 'validation')):
            data = group[field]
            x = np.asarray(group['epoch'])
            y = np.asarray(data['mean'])
            lo, hi = np.asarray(data['ci_lower']), np.asarray(data['ci_upper'])
            line, = ax.plot(x, y, label=method)
            mask = np.isfinite(lo) & np.isfinite(hi)
            if scale == 'log': mask &= (lo > 0) & (hi > 0)
            if mask.any():
                ax.fill_between(x, lo, hi, where=mask, alpha=.18, color=line.get_color())
    for ax, field in zip(axes, ('train', 'validation')):
        ax.set(xlabel='Epoch', ylabel=metric.upper(), title=field.capitalize())
        ax.set_yscale(scale)
        ax.grid(True, alpha=.2)
        ax.legend(loc="upper right")
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return curves
