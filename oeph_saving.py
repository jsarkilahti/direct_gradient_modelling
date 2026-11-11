"""Serialize experiment configuration, metrics, predictions, and model checkpoints."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import numpy as np
import torch


def jsonable(value):
    """Handle jsonable for the experiment workflow."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(destination, value):
    """Save json to the experiment output."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(jsonable(value), indent=2, allow_nan=True))


def save_configuration(configuration, directory):
    """Save configuration to the experiment output."""
    write_json(Path(directory) / 'configuration.json', configuration)


def save_dataset(split, metadata, directory, variant_index):
    """Compatibility entry point: datasets live exclusively in the shared data cache."""
    raise RuntimeError('Use load_or_prepare_dataset for shared data persistence')


def save_checkpoint(state, directory, index):
    """Save a model checkpoint for later evaluation or reconstruction."""
    destination = Path(directory) / 'checkpoints' / f'run_{index if isinstance(index, str) else f"{index:04d}"}.pt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if state is None:
        raise ValueError('Best model state is unavailable; refusing to save an empty checkpoint')
    torch.save({key: tensor.detach().cpu().clone() for key, tensor in state.items()}, destination)
    return str(destination.relative_to(directory))


def save_predictions(predictions, targets, directory, index):
    """Save prediction arrays for result reconstruction."""
    destination = Path(directory) / 'predictions' / f'run_{index if isinstance(index, str) else f"{index:04d}"}.npz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(predictions) != len(targets):
        raise ValueError('Prediction/target trajectory counts differ')
    arrays = {}
    for j, (prediction, target) in enumerate(zip(predictions, targets)):
        arrays[f'prediction_{j}'] = np.asarray(prediction)
        arrays[f'target_{j}'] = np.asarray(target)
    np.savez_compressed(destination, **arrays)
    return str(destination.relative_to(directory))


def load_predictions(directory, relative_path):
    """Load predictions for experiment processing."""
    with np.load(Path(directory) / relative_path, allow_pickle=False) as arrays:
        ids = sorted(int(k.split('_')[1]) for k in arrays.files if k.startswith('prediction_'))
        return ([arrays[f'prediction_{j}'].copy() for j in ids],
                [arrays[f'target_{j}'].copy() for j in ids])


def save_continuation_checkpoint(checkpoint, directory, index):
    """Persist model, optimizer, scheduler, and continuation metadata without lossy JSON."""
    destination = Path(directory) / 'checkpoints' / f'continuation_{index if isinstance(index, str) else f"{index:04d}"}.pt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        raise ValueError('Missing continuation checkpoint')
    torch.save(checkpoint, destination)
    return str(destination.relative_to(directory))


# Dataset cache contains full time series, never training windows.
def dataset_cache_key(config):
    """Hash data-affecting settings and the CT source file contents."""
    from dataclasses import asdict
    import hashlib
    payload = {
        'data': jsonable({k:v for k,v in asdict(config.data).items() if k != 'normalization'}),
        'model_sampling_time': config.model.model_sampling_time if config.data.system == 'CT' else None,
        'format_version': 2,
    }
    if config.data.system == 'CT':
        candidates = sorted(Path(config.data.data_root).rglob(config.data.dataset_filename))
        if len(candidates) != 1:
            raise ValueError(f'Expected one CT data source, found {len(candidates)}')
        payload['source_sha256'] = hashlib.sha256(candidates[0].read_bytes()).hexdigest()
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_or_prepare_dataset(config, cache_root):
    """Reuse matching time-series data; construct windows separately after loading."""
    from oeph_definitions import SplitDataset, TimeSeriesDataset
    from oeph_data import prepare_data
    import os
    key = dataset_cache_key(config)
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    folder = cache / key
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'dataset.npz'
    manifest = folder / 'metadata.json'
    if path.exists() and manifest.exists():
        try:
            info = json.loads(manifest.read_text())
            if info.get('key') != key or info.get('format_version') != 2:
                raise ValueError('Cache manifest mismatch')
            with np.load(path, allow_pickle=False) as data:
                splits = {}
                for name in ('train', 'validation', 'test'):
                    fields = {}
                    for field in ('controls', 'outputs', 'states', 'states_ph', 'time', 'sample_index', 'series_ids'):
                        array_key = f'{name}_{field}'
                        fields[field] = data[array_key].copy() if array_key in data else None
                    fields['Ts'] = info['splits'][name]['Ts']
                    fields['metadata'] = info['splits'][name]['metadata']
                    splits[name] = TimeSeriesDataset(**fields)
            print(f'[DATA] LOAD successful: {path}', flush=True)
            return SplitDataset(**splits), info['metadata'], path, True
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            print(f'[DATA] Cache invalid ({exc}); regenerating', flush=True)
    print(f'[DATA] GENERATE/PREPARE system={config.data.system} key={key[:12]}', flush=True)
    from dataclasses import replace
    from oeph_definitions import NormalizationConfig
    # Persist the untransformed time series. Method-specific transforms happen
    # after cache loading and are never written to the cache.
    raw_config = replace(config, data=replace(config.data,
        normalization=NormalizationConfig(enabled=False)),
        training=replace(config.training, square_outputs=False),
        model=replace(config.model, dtype="float64") if config.data.system == "CT" else config.model)
    split, metadata = prepare_data(raw_config)
    arrays = {}
    details = {}
    for name in ('train', 'validation', 'test'):
        ds = getattr(split, name)
        details[name] = {'Ts': ds.Ts, 'metadata': jsonable(ds.metadata)}
        for field in ('controls', 'outputs', 'states', 'states_ph', 'time', 'sample_index', 'series_ids'):
            value = getattr(ds, field, None)
            if value is not None:
                arrays[f'{name}_{field}'] = np.asarray(value)
    import tempfile
    with tempfile.NamedTemporaryFile(dir=cache, suffix='.npz', delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(manifest, {'key': key, 'format_version': 2, 'metadata': metadata, 'splits': details, 'configuration': jsonable(config.data)})
    print(f'[DATA] GENERATION successful; cached: {path}', flush=True)
    return split, metadata, path, False
