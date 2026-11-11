"""Summarize experiment results, workloads, and timing measurements in text reports."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from oeph_definitions import RunResult, StudyResult
from oeph_stats import compute_statistics, paired_method_differences
from oeph_plotting import plot_learning_curves, plot_comparison, plot_test_trajectory
from oeph_representatives import select_representatives
from oeph_paper_figures import plot_paper_curves
from types import SimpleNamespace


from oeph_saving import jsonable as _jsonable, write_json, save_configuration, save_checkpoint, save_predictions, load_predictions, save_continuation_checkpoint


def _run_identifier(run):
    """Handle run identifier for the experiment workflow. Used to assemble paired and individual experiment reports."""
    from oeph_training import _level_folder
    return f'{run.method}_seed{run.model_seed}_shuffle{run.shuffle_seed}'


def _save_level(study, path, *, figures, options):
    """Save level to the experiment output. Used to assemble paired and individual experiment reports."""
    path.mkdir(parents=True, exist_ok=True)
    records = []
    for i, run in enumerate(study.runs):
        record = {k: _jsonable(v) for k, v in asdict(run).items()}
        # Persist only independent scalar errors. RMSE/NRMSE are derived from
        # MSE and the denominator's population standard deviation.
        record['metrics'] = {k: v for k, v in record['metrics'].items()
                             if not (k in ('rmse', 'nrmse') or k.endswith('_rmse')
                                     or k.endswith('_nrmse'))}
        record['learning_rates'] = _jsonable(getattr(run, 'learning_rates', []))
        record['training_metadata'] = _jsonable(getattr(run, 'training_metadata', {}))
        record['trajectory_metrics'] = _jsonable(getattr(run, 'trajectory_metrics', []))
        record['segment_metrics'] = _jsonable(getattr(run, 'segment_metrics', []))
        record['trajectory_status'] = _jsonable(getattr(run, 'trajectory_status', []))
        identifier = _run_identifier(run)
        if hasattr(run, 'predictions') and hasattr(run, 'targets'):
            record['predictions_path'] = save_predictions(run.predictions, run.targets, path, identifier)
        if getattr(run, 'best_model_state', None) is not None:
            destination = path / 'checkpoints' / f'run_{identifier}.pt'
            if not destination.exists():
                record['checkpoint_path'] = save_checkpoint(run.best_model_state, path, identifier)
            else:
                record['checkpoint_path'] = str(destination.relative_to(path))
        if getattr(run, 'continuation_checkpoint', None) is not None:
            record['continuation_checkpoint_path'] = save_continuation_checkpoint(run.continuation_checkpoint, path, identifier)
        records.append(record)
    write_json(path / 'runs.json', records)
    stats = compute_statistics(study)
    write_json(path / 'statistics.json', stats)
    write_json(path / 'paired_statistics.json', paired_method_differences(study))
    write_json(path / 'representatives.json', select_representatives(study, test_metric=options.get('test_metric', 'nrmse')))
    write_json(path / 'plot_configuration.json', options)
    write_report(study.runs, path / 'report_paired.txt', title='PAIRED EXPERIMENT REPORT', configuration=study.configuration)
    for method in sorted(set(r.method for r in study.runs)):
        write_report([r for r in study.runs if r.method == method],
                     path / f'report_individual_{method}.txt',
                     title=f'MODEL EXPERIMENT REPORT: {method}', configuration=study.configuration)
    if figures:
        from oeph_plotting import plot_experiment_figures
        plot_experiment_figures(study, path / 'figures', options)
    return path


def save_study(study, directory, *, figures=True, plot_options=None):
    """Persist each condition separately, with a shared top-level configuration."""
    from oeph_training import _level_folder
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if study.configuration is not None:
        save_configuration(study.configuration, path)
    options = plot_options or {}
    by_level = {}
    for run in study.runs:
        level = run.configuration.get('level')
        name = _level_folder(run.configuration.get('study', 'baseline'), level)
        by_level.setdefault(name, []).append(run)
    for name, runs in by_level.items():
        _save_level(StudyResult(configuration=study.configuration, runs=runs),
                    path / name, figures=figures, options=options)
    return path


def reconstruct_results(directory, *, figures=True, plot_options=None):
    """Rebuild statistics, reports and figures from disk; never train or overwrite checkpoints."""
    path = Path(directory)
    records = json.loads((path / 'runs.json').read_text())
    configuration_file = path / 'configuration.json'
    configuration = json.loads(configuration_file.read_text()) if configuration_file.exists() else None
    study = StudyResult(configuration=configuration)
    for record in records:
        run = RunResult(**{k: record[k] for k in RunResult.__dataclass_fields__ if k in record})
        from oeph_stats import restore_derived_metrics
        run.metrics = restore_derived_metrics(run.metrics)
        run.learning_rates = record.get('learning_rates', [])
        run.training_metadata = record.get('training_metadata', {})
        run.trajectory_metrics = record.get('trajectory_metrics', [])
        run.segment_metrics = record.get('segment_metrics', [])
        run.trajectory_status = record.get('trajectory_status', [])
        relative = record.get('predictions_path', record.get('predictions_file'))
        if relative:
            run.predictions, run.targets = load_predictions(path, relative)
        study.runs.append(run)
    # Reconstruction must never reserialize run records or overwrite saved checkpoints.
    # Only regenerate derived statistics, report and figures.
    original_runs = (path / 'runs.json').read_bytes()
    checkpoint_bytes = {p: p.read_bytes() for p in (path / 'checkpoints').glob('*.pt')} if (path / 'checkpoints').exists() else {}
    _save_level(study, path, figures=figures, options=plot_options or {})
    (path / 'runs.json').write_bytes(original_runs)
    for checkpoint, content in checkpoint_bytes.items():
        if not checkpoint.exists() or checkpoint.read_bytes() != content:
            raise RuntimeError(f'Reconstruction changed a checkpoint: {checkpoint}')
    return study


"""Human-readable per-condition reports derived from recorded runs."""
from collections import defaultdict
import numpy as np

METRICS = ('mse', 'rmse', 'nrmse')
# Timings are grouped by workload rather than emitted as an unlabelled metric dump.
TRAINING_TIMINGS = (
    ('training_seconds', 'Total training wall time', 's'),
    ('timing_forward_steady_mean_seconds', 'Forward, per epoch (steady state)', 's/epoch'),
    ('timing_backward_steady_mean_seconds', 'Backward, per epoch (steady state)', 's/epoch'),
    ('timing_optimizer_steady_mean_seconds', 'Optimizer, per epoch (steady state)', 's/epoch'),
    ('timing_validation_steady_mean_seconds', 'Validation, per epoch (steady state)', 's/epoch'),
    ('timing_total_steady_mean_seconds', 'Total, per epoch (steady state)', 's/epoch'),
)
EVALUATION_TIMINGS = (
    ('checkpoint_metrics_seconds', 'All checkpoint metric evaluations', 's'),
    ('training_windowed_evaluation_seconds', 'Training windowed evaluation', 's'),
    ('validation_windowed_evaluation_seconds', 'Validation windowed evaluation', 's'),
    ('training_full_rollout_evaluation_seconds', 'Training full rollout evaluation', 's'),
    ('validation_full_rollout_evaluation_seconds', 'Validation full rollout evaluation', 's'),
    ('testing_seconds', 'Test rollout evaluation', 's'),
)
HAMILTONIAN_TIMINGS = (
    ('hamiltonian_potential_forward_ms', 'Potential forward', 'ms/state'),
    ('hamiltonian_gradient_ms', 'Gradient evaluation', 'ms/state'),
    ('hamiltonian_total_block_ms', 'Hamiltonian block evaluation', 'ms/state'),
)


def _config_section(configuration, section):
    """Handle config section for the experiment workflow. Used to assemble paired and individual experiment reports."""
    if configuration is None:
        return None
    if isinstance(configuration, dict):
        return configuration.get(section)
    return getattr(configuration, section, None)


def _config_value(configuration, section, key):
    """Handle config value for the experiment workflow. Used to assemble paired and individual experiment reports."""
    obj = _config_section(configuration, section)
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _constant(values):
    """Format a configuration value that does not vary across runs."""
    values = [v for v in values if v is not None]
    if not values:
        return 'not recorded'
    unique = {str(v) for v in values}
    return str(values[0]) if len(unique) == 1 else 'varies: ' + ', '.join(sorted(unique))


def _timing_rows(lines, group, fields):
    """Format workload-aware timing metrics for experiment reports."""
    for key, label, unit in fields:
        values = [r.metrics.get(key) for r in group]
        if any(v is not None for v in values):
            lines.append(f'  {label} [{unit}]: {_summary(values)}')


def _number(value):
    """Handle number for the experiment workflow. Used to assemble paired and individual experiment reports."""
    try:
        x = float(value)
        return f'{x:.8g}' if np.isfinite(x) else 'not available'
    except (TypeError, ValueError):
        return 'not available'


def _summary(values):
    """Summarize varying run metrics across experiment seeds."""
    a = np.asarray([float(x) for x in values if x is not None], dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return 'not available (not measured)'
    return ('best=' + _number(np.min(a)) + '  mean=' + _number(np.mean(a))
            + '  median=' + _number(np.median(a))
            + '  Q1=' + _number(np.percentile(a,25))
            + '  Q3=' + _number(np.percentile(a,75))
            + f'  n={len(a)}')


def _value(run, split, metric):
    """Read only metrics evaluated at the same selected best checkpoint."""
    key = f'{split}_{metric}'
    if split == 'testing':
        return run.metrics.get(key, run.metrics.get(metric))
    return run.metrics.get(key)


def _best(runs, key):
    """Select the best run according to the requested evaluation metric."""
    eligible = [(r, r.metrics.get(key)) for r in runs]
    eligible = [(r, float(v)) for r, v in eligible if v is not None and np.isfinite(v)]
    if not eligible: return 'not available'
    r, v = min(eligible, key=lambda item: item[1])
    return f'model_seed={r.model_seed} shuffle_seed={r.shuffle_seed}  {key}={_number(v)}'


def write_report(runs, destination, *, title, configuration=None):
    """Write a text report describing an experiment and its results."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, '=' * len(title), '',
             f'Number of runs: {len(runs)}',
             f'Number of distinct model seeds: {len(set(r.model_seed for r in runs))}',
             'Model seeds: ' + ', '.join(map(str, sorted(set(r.model_seed for r in runs)))),
             'Model/shuffle seed pairs: ' + ', '.join(f'({r.model_seed}, {r.shuffle_seed})' for r in runs),
             'Methods: ' + ', '.join(sorted(set(r.method for r in runs))), '',
             'Best run by validation loss: ' + _best(runs, 'best_validation_loss'),
             'Best run by test MSE: ' + _best(runs, 'mse'),
             'Best run by test RMSE: ' + _best(runs, 'rmse'),
             'Best run by test NRMSE: ' + _best(runs, 'nrmse'), '']
    groups = defaultdict(list)
    for r in runs: groups[r.method].append(r)
    for method, group in sorted(groups.items()):
        lines.extend([f'MODEL: {method}', '-' * (7 + len(method)),
                      f'Runs: {len(group)}',
                      'Seeds: ' + ', '.join(map(str, sorted(set(r.model_seed for r in group)))),
                      'Best validation: ' + _best(group, 'best_validation_loss'),
                      'Best test: ' + _best(group, 'mse'), '',
                      'Neural architecture:'])
        architectures = {str((getattr(r, 'training_metadata', {}) or {}).get('model_architecture', 'not recorded')) for r in group}
        for architecture in sorted(architectures):
            lines.extend('  ' + line for line in architecture.splitlines())
        lines.extend(['', 'Parameter counts:'])
        for key in ('parameter_count','trainable_parameter_count',
                    'hamiltonian_parameter_count','hamiltonian_trainable_parameter_count'):
            lines.append(f'  {key}: {_constant([r.metrics.get(key) for r in group])}')
        for split in ('training_windowed','training_full_rollout','validation_windowed','validation_full_rollout','testing'):
            lines.extend(['', split.upper().replace('_', ' ') + ' (selected best checkpoint):'])
            for metric in METRICS:
                lines.append(f'  {metric.upper():6s} {_summary([_value(r, split, metric) for r in group])}')
        lines.extend(['', 'TIMING WORKLOAD:',
                      f'  Training minibatch size [windows/batch]: {_config_value(configuration, "training", "batch_size") or "not recorded"}',
                      f'  Training windows [windows]: {_constant([r.configuration.get("training_windows") for r in group])}',
                      f'  Prediction horizon [samples/window]: {_constant([r.configuration.get("prediction_horizon") for r in group])}',
                      f'  History length [samples/window]: {_config_value(configuration, "windows", "history_length") or "not recorded"}',
                      f'  Training epochs completed: {_constant([(getattr(r, "training_metadata", {}) or {}).get("executed_epochs") for r in group])}',
                      f'  Epoch timing measurements: {_constant([r.metrics.get("timing_measured_epochs") for r in group])}',
                      f'  Steady-state epochs (first excluded): {_constant([r.metrics.get("timing_steady_state_epochs") for r in group])}',
                      f'  Windowed training evaluation size [windows]: {_constant([r.configuration.get("training_windows") for r in group])}',
                      '  Full-rollout and test evaluations: complete configured split; durations are totals, not per-sample timings.',
                      '', 'TRAINING TIMING (across seeds):'])
        _timing_rows(lines, group, TRAINING_TIMINGS)
        lines.extend(['', 'CHECKPOINT AND TEST EVALUATION TIMING (across seeds):'])
        _timing_rows(lines, group, EVALUATION_TIMINGS)
        lines.extend(['', 'HAMILTONIAN BENCHMARK:',
                      f'  Probe batch size [states/call]: {_constant([r.metrics.get("hamiltonian_probe_batch_size") for r in group])}',
                      f'  Timed repetitions [calls]: {_constant([r.metrics.get("hamiltonian_probe_repetitions") for r in group])}',
                      f'  Warm-up repetitions [calls]: {_constant([r.metrics.get("hamiltonian_probe_warmup") for r in group])}',
                      '  Excludes encoder, PH dynamics, and numerical integration.',
                      '  Hamiltonian block includes gradient evaluation; do not add potential and gradient timings.'])
        _timing_rows(lines, group, HAMILTONIAN_TIMINGS)
        lines.append('')
    lines.extend(['Notes:',
        'All reported split metrics are measured at the selected best-validation checkpoint.',
        'Training and validation each include separate windowed and full-rollout evaluation.',
        'NRMSE uses the configured denominator and the population standard deviation of target values.',
        'Hamiltonian timing uses a separate post-training single-state probe, excluding encoder, integration and PH dynamics.',
        'Hamiltonian total block timing equals the measured gradient-call timing (which includes the potential for Autograd).',
        'Timing results are machine- and device-dependent.', ''])
    destination.write_text('\n'.join(lines), encoding='utf-8')
    return destination
