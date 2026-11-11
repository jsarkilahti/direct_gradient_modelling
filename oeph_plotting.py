"""Plot training histories, confidence intervals, and test rollouts for individual and paired experiments."""
from __future__ import annotations
import numpy as np


def _pyplot():
    """Initialize plotting support for experiment figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _metric_history(history, metric):
    """Extract an epoch-indexed metric history for plotting."""
    a = np.asarray(history, dtype=float)
    if metric == 'mse': return a
    if metric == 'rmse': return np.sqrt(np.maximum(a, 0))
    raise ValueError('Training histories contain MSE; NRMSE requires per-epoch denominators that are not saved')


def _visible(x, y, scale):
    """Handle visible for the experiment workflow. Used to visualize learning curves and saved test trajectories."""
    mask = np.isfinite(y)
    if scale == 'log': mask &= y > 0
    return x[mask], y[mask]


def plot_learning_curves(study, destination, *, scale='linear', metric='mse', aggregate=False,
                         confidence=0.95, show_individual=True):
    """Plot the recorded training and validation losses."""
    from oeph_stats import summarize
    from collections import defaultdict
    if scale not in ('linear', 'log'): raise ValueError('scale must be linear or log')
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    groups = defaultdict(list)
    for run in study.runs:
        groups[(run.method, run.configuration.get('level'))].append(run)
    for (method, level), runs in groups.items():
        for ax, attr in zip(axes, ('training_loss', 'validation_loss')):
            curves = [_metric_history(getattr(run, attr), metric) for run in runs]
            if show_individual:
                for run, curve in zip(runs, curves):
                    x, y = _visible(np.arange(1, len(curve)+1), curve, scale)
                    ax.plot(x, y, alpha=.25 if aggregate else .75,
                            label=f'{method} {level} seed={run.model_seed}' if not aggregate else None)
            if aggregate and curves:
                n = max(map(len, curves))
                summary = [summarize([c[i] if i < len(c) else np.nan for c in curves], confidence) for i in range(n)]
                x = np.arange(1, n+1)
                y = np.array([s['mean'] for s in summary])
                lo = np.array([s['ci_lower'] for s in summary])
                hi = np.array([s['ci_upper'] for s in summary])
                xx, yy = _visible(x, y, scale)
                line, = ax.plot(xx, yy, label=f'{method} {level} mean', linewidth=2)
                mask = np.isfinite(lo) & np.isfinite(hi)
                if scale == 'log': mask &= (lo > 0) & (hi > 0)
                if len(runs) > 1 and mask.any():
                    ax.fill_between(x, lo, hi, where=mask, color=line.get_color(), alpha=.18)
    for ax, name in zip(axes, ('Training', 'Validation')):
        ax.set(xlabel='Epoch', ylabel=f'{metric.upper()} (training coordinates)', title=name)
        ax.set_yscale(scale)
        if ax.lines: ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=.2)
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def plot_comparison(study, destination, *, metric='nrmse', scale='linear', confidence=0.95):
    """Compare experiment methods using their recorded results."""
    from oeph_stats import compute_statistics
    if scale not in ('linear', 'log'): raise ValueError('scale must be linear or log')
    stats = compute_statistics(study, confidence)
    keys = list(stats)
    values = [stats[k]['metrics'].get(metric, {}).get('mean', np.nan) for k in keys]
    plt = _pyplot(); fig, ax = plt.subplots(figsize=(max(6, len(keys)*1.2), 4))
    ax.bar(range(len(keys)), values)
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha='right')
    ax.set(ylabel=f'Test {metric.upper()}', title='Model comparison')
    ax.set_yscale(scale)
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def plot_test_trajectory(run, destination, index=0, *, sampling_time=None, scale='linear'):
    """Plot a saved model rollout against the reference trajectory."""
    if scale not in ('linear', 'log'): raise ValueError('scale must be linear or log')
    if not hasattr(run, 'predictions') or index >= len(run.predictions):
        raise ValueError('Saved predictions unavailable for requested trajectory')
    prediction = np.asarray(run.predictions[index])
    target = np.asarray(run.targets[index])
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target shapes differ')
    if prediction.ndim == 1:
        prediction = prediction[:, None]; target = target[:, None]
    elif prediction.ndim != 2:
        raise ValueError('Trajectory arrays must have shape [samples, channels]')
    x = np.arange(len(target)) * (1 if sampling_time is None else sampling_time)
    plt = _pyplot(); fig, ax = plt.subplots()
    for j in range(target.shape[1]):
        ax.plot(x, target[:, j], label=f'Measured {j}')
        ax.plot(x, prediction[:, j], '--', label=f'Predicted {j}')
    ax.set(xlabel='Sample index' if sampling_time is None else 'Time (s)',
           ylabel=run.configuration.get('output_unit', 'Output (model coordinates)'),
           title=f'{run.method} test trajectory {index}')
    ax.set_yscale(scale); ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def paper_learning_curve_intervals(runs, *, system, scale, metric='nrmse', confidence=0.95,
                                   train_target_std=None, validation_target_std=None,
                                   ct_output_scale=None):
    """Reference paper convention: transform each MSE history before aggregation.

    Linear: arithmetic mean and Student-t confidence interval.
    Log: geometric mean and Student-t interval in log space.
    Input runs need training_loss_history, validation_loss_history and experiment_name.
    CT NRMSE uses per-run ct_output_scale; MSD NRMSE uses reference target std.
    """
    from scipy.stats import t
    if system not in ('MSD', 'CT') or scale not in ('linear', 'log') or metric.lower() not in ('mse','rmse','nrmse'):
        raise ValueError('Invalid paper curve configuration')
    if not 0 < confidence < 1: raise ValueError('Invalid confidence')
    output = {}
    for method in sorted({str(r.experiment_name) for r in runs}):
        chosen = [r for r in runs if str(r.experiment_name) == method]
        curves = {}
        for field, attr, std in (('train','training_loss_history',train_target_std),
                                 ('validation','validation_loss_history',validation_target_std)):
            transformed_histories = []
            for run in chosen:
                values = np.asarray(getattr(run,attr),dtype=float).ravel()
                if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError('Invalid MSE history')
                if metric.lower() != 'mse': values = np.sqrt(values)
                if metric.lower() == 'nrmse':
                    if std is None or std <= 0: raise ValueError('Reference target std required')
                    if system == 'CT':
                        factor = getattr(run,'ct_output_scale',ct_output_scale)
                        if factor is None: raise ValueError('CT output scale required')
                        values = values * float(factor)
                    values = values / float(std)
                transformed_histories.append(values)
            if len({len(a) for a in transformed_histories}) != 1:
                raise ValueError('Unequal history lengths')
            values = np.stack(transformed_histories)
            if scale == 'log':
                if np.any(values <= 0): raise ValueError('Log scale requires positive values')
                values = np.log(values)
            center = values.mean(axis=0)
            n = len(values)
            if n > 1:
                margin = t.ppf((1+confidence)/2,df=n-1) * values.std(axis=0,ddof=1)/np.sqrt(n)
            else:
                margin = np.full_like(center,np.nan)
            lo,hi = center-margin,center+margin
            if scale == 'log': center,lo,hi=np.exp(center),np.exp(lo),np.exp(hi)
            curves[field] = {'mean':center,'ci_lower':lo,'ci_upper':hi,'n':n}
        curves['epoch']=np.arange(1,len(curves['train']['mean'])+1)
        output[method]=curves
    return output


def plot_experiment_figures(study, destination, options):
    """Paired/per-method paper presentation, using saved chronological histories."""
    from pathlib import Path
    from collections import defaultdict
    from scipy.stats import t
    plt = _pyplot()
    out = Path(destination)
    groups = defaultdict(list)
    for run in study.runs:
        groups[run.method].append(run)
    if not groups:
        return
    system = study.runs[0].configuration.get('system', 'CT')
    formats = tuple(options.get('formats', ('png',)))
    scales = tuple(options.get('scales', ('linear', 'log')))
    metric = options.get('training_metric', 'mse').lower()
    confidence = float(options.get('confidence', .95))
    if not 0 < confidence < 1:
        raise ValueError('confidence must lie strictly between 0 and 1')
    alpha = float(options.get('individual_run_alpha', .40))
    if not 0 <= alpha <= 1:
        raise ValueError('individual_run_alpha must lie in [0,1]')
    colors = options.get('model_colors', {})
    train_labels = options.get('train_axis_labels', {}).get(system, {})
    rollout_labels = options.get('rollout_axis_labels', {}).get(system, {})
    dpi = int(options.get('figure_dpi', 180))
    def label(value, fallback):
        """Handle label for the experiment workflow. Used to visualize learning curves and saved test trajectories."""
        return fallback if value is None else value
    def apply_style():
        """Handle apply style for the experiment workflow. Used to visualize learning curves and saved test trajectories."""
        plt.rcParams.update({'axes.labelsize': options.get('axis_label_fontsize',16),
            'xtick.labelsize':options.get('tick_label_fontsize',16),
            'ytick.labelsize':options.get('tick_label_fontsize',16),
            'axes.titlesize':options.get('title_fontsize',16)})
    apply_style()
    def history_metric(run, field):
        """Select the requested training-history metric for plotting."""
        values = np.asarray(getattr(run, field), dtype=float)
        if metric == 'mse': return values
        if metric in ('rmse', 'nrmse'):
            values = np.sqrt(np.maximum(values,0))
            if metric == 'nrmse':
                split = 'training' if field == 'training_loss' else 'validation'
                std = run.metrics.get(f'{split}_windowed_target_std')
                if std is None or not np.isfinite(std) or std <= 0:
                    raise ValueError(f'NRMSE learning curves require {split}_windowed_target_std')
                values = values / std
            return values
        raise ValueError('training_metric must be mse, rmse, or nrmse')
    def curve_plot(selected, dest, scale, show_runs):
        """Draw an aggregate learning curve and its confidence interval."""
        fig, axes = plt.subplots(1,2,figsize=options.get('train_figure_shape',(12,5)),sharex=True)
        for method, runs in selected.items():
            for ax, field in zip(axes, ('training_loss','validation_loss')):
                histories = [history_metric(run,field) for run in runs]
                if not histories: continue
                n_epochs = max(map(len,histories))
                if n_epochs == 0: continue
                data = np.full((len(histories),n_epochs),np.nan)
                for i,h in enumerate(histories): data[i,:len(h)] = h
                if scale == 'log':
                    data = np.where(data>0,np.log(data),np.nan)
                x = np.arange(1,n_epochs+1)
                color = colors.get(method)
                if show_runs and options.get('show_individual_runs',True):
                    for row in data:
                        ax.plot(x,np.exp(row) if scale=='log' else row,
                                color=color,alpha=alpha,linewidth=1)
                counts = np.isfinite(data).sum(axis=0)
                with np.errstate(invalid='ignore',divide='ignore'):
                    center = np.nanmean(data,axis=0)
                    sd = np.nanstd(data,axis=0,ddof=1)
                    margin = t.ppf((1+confidence)/2,counts-1)*sd/np.sqrt(counts)
                lo,hi=center-margin,center+margin
                if scale=='log': center,lo,hi=np.exp(center),np.exp(lo),np.exp(hi)
                line,=ax.plot(x,center,color=color,linewidth=2,label=f'{method} (n={len(runs)})')
                mask=np.isfinite(lo)&np.isfinite(hi)&(counts>1)
                if mask.any(): ax.fill_between(x,lo,hi,where=mask,color=line.get_color(),alpha=.18)
        for ax, field, name in zip(axes,('train_y','validation_y'),('Training','Validation')):
            ax.set_xlabel(label(train_labels.get('x'),'Epoch'))
            ylabel=train_labels.get(field)
            if ylabel is not None: ax.set_ylabel(ylabel)
            else: ax.set_ylabel(metric.upper())
            ax.set_yscale(scale)
            if options.get('train_xlimits') is not None: ax.set_xlim(options['train_xlimits'])
            if options.get('train_ylimits') is not None: ax.set_ylim(options['train_ylimits'])
            if options.get('msd_mark_warmup',False) and system=='MSD':
                boundary=options.get('msd_warmup_end_epoch')
                if boundary is not None: ax.axvline(boundary,color='gray',linestyle=':')
            ax.grid(True,alpha=.2)
            if ax.lines: ax.legend(loc=options.get('train_legend_location','upper right'),fontsize=options.get('train_legend_fontsize',16))
        if options.get('train_title') is not None: fig.suptitle(options['train_title'])
        fig.tight_layout();dest.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(dest,dpi=dpi);plt.close(fig)
    def best_run(runs):
        """Select the run used to illustrate a method in the figures."""
        eligible=[r for r in runs if np.isfinite(r.metrics.get('best_validation_loss',np.nan))]
        return min(eligible,key=lambda r:r.metrics['best_validation_loss']) if eligible else None
    def test_plot(selected,dest):
        """Draw a test rollout and the corresponding reference output."""
        chosen={name:best_run(runs) for name,runs in selected.items()}
        chosen={name:r for name,r in chosen.items() if r is not None and hasattr(r,'predictions')}
        if not chosen: return
        count=max(len(r.predictions) for r in chosen.values())
        chunk=options.get('msd_rollout_chunk_size',1000) if system=='MSD' else None
        for index in range(count):
            present={name:r for name,r in chosen.items() if index<len(r.predictions)}
            if not present: continue
            length=min(len(r.targets[index]) for r in present.values())
            step=chunk or length
            if step<=0: continue
            for start in range(0,length,step):
                stop=min(start+step,length)
                fig,ax=plt.subplots(figsize=options.get('rollout_figure_shape',(10,5)))
                first=next(iter(present.values()))
                target=np.asarray(first.targets[index])[start:stop]
                if target.ndim==1:target=target[:,None]
                sample_time=first.configuration.get('sampling_time')
                if sample_time is None: sample_time=4.0 if system=='CT' else .1
                x=np.arange(start,stop)*float(sample_time)
                for j in range(target.shape[1]):
                    ax.plot(x,target[:,j],color=options.get('ground_truth_color','black'),
                            linestyle=options.get('ground_truth_linestyle','--'),label=f'Measured {j}')
                for name,run in present.items():
                    pred=np.asarray(run.predictions[index])[start:stop]
                    if pred.ndim==1:pred=pred[:,None]
                    for j in range(pred.shape[1]):
                        ax.plot(x,pred[:,j],color=colors.get(name),label=f'{name} {j}')
                ax.set_xlabel(label(rollout_labels.get('x'),'Time (s)'))
                ax.set_ylabel(label(rollout_labels.get('y'),'Output'))
                if options.get('rollout_title') is not None:ax.set_title(options['rollout_title'])
                ylim=options.get('rollout_ylimits',{}).get(system)
                if ylim is not None:ax.set_ylim(ylim)
                if system=='CT' and options.get('ct_time_limits') is not None:
                    ax.set_xlim(options['ct_time_limits'])
                ax.grid(True,alpha=.2)
                ax.legend(loc=options.get('rollout_legend_location','upper right'),fontsize=options.get('rollout_legend_fontsize',16))
                fig.tight_layout()
                path=dest.parent/f'{dest.stem}_trajectory{index:02d}_samples{start:06d}-{stop:06d}{dest.suffix}'
                path.parent.mkdir(parents=True,exist_ok=True)
                fig.savefig(path,dpi=dpi);plt.close(fig)
    destinations = []
    if options.get('make_combined', True): destinations.append(('paired',groups))
    if options.get('make_individual', True):
        destinations.extend((f'individual/{m}',{m:r}) for m,r in groups.items())
    for folder,selected in destinations:
        for fmt in formats:
            if fmt not in ('png','svg'):raise ValueError('Invalid plot format')
            if options.get('make_convergence', True):
                for scale in scales:
                    if scale not in ('linear','log'):raise ValueError('Invalid plot scale')
                    curve_plot(selected,out/folder/f'all_CI_{scale}.{fmt}',scale,True)
                    if options.get('make_mean_ci', True):
                        curve_plot(selected,out/folder/f'mean_CI_{scale}.{fmt}',scale,False)
            test_plot(selected,out/folder/f'test_best.{fmt}')
