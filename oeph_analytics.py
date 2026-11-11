"""Measure model parameter counts, elapsed time, and execution timings used in experiment comparisons."""
from __future__ import annotations
from contextlib import contextmanager
from time import perf_counter


def parameter_counts(model):
    """Count total and trainable model parameters for experiment reports."""
    parameters = list(model.parameters())
    return {'total': sum(p.numel() for p in parameters),
            'trainable': sum(p.numel() for p in parameters if p.requires_grad)}


@contextmanager
def elapsed_time(destination, key):
    """Return elapsed wall-clock time for an experiment operation."""
    start = perf_counter()
    try:
        yield
    finally:
        destination[key] = perf_counter() - start


def hamiltonian_parameter_counts(model):
    """Reference CT inventory, including allocated/unused MLP output heads."""
    h = model.hamiltonian_component
    inventory = {name: {'shape':list(p.shape),'count':p.numel(),'trainable':p.requires_grad}
                 for name,p in h.named_parameters()}
    result = {'total_parameters':sum(v['count'] for v in inventory.values()),
              'trainable_parameters':sum(v['count'] for v in inventory.values() if v['trainable']),
              'parameter_inventory':inventory}
    network=getattr(h,'network',None)
    if hasattr(network,'net') and hasattr(network.net[-1],'out_features'):
        heads=network.net[-1].out_features
        result['allocated_output_heads']=heads
        result['used_output_heads']=1
        result['unused_output_head_parameters']=(heads-1)*(network.net[-1].in_features+1)
    return result


def benchmark_hamiltonian(model, *, repetitions=20, warmup=5, seed=0):
    """Post-training single-state Hamiltonian probe, as in the original benchmark.

    Separate potential forward and gradient calls. Gradient-network potentials may
    be unavailable; in that case potential time is not reported. This probe does
    not include the encoder, PH dynamics or numerical integration.
    """
    import torch
    import numpy as np
    parameter = next(model.parameters())
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(seed)
    x = torch.randn((1, model.dim_x), generator=generator,
                    device=parameter.device, dtype=parameter.dtype) * 0.1
    was_training = model.training
    model.eval()
    def synchronize():
        """Synchronize device execution before timing measurements."""
        if parameter.device.type == 'cuda': torch.cuda.synchronize(parameter.device)
        elif parameter.device.type == 'mps': torch.mps.synchronize()
    def measure(fn):
        """Measure elapsed execution time for the requested operation."""
        times = []
        for i in range(warmup + repetitions):
            synchronize()
            start = perf_counter()
            with torch.enable_grad():
                result = fn(x.detach().clone().requires_grad_(True))
            synchronize()
            if i >= warmup: times.append((perf_counter()-start)*1000)
            del result
        return float(np.mean(times))
    try:
        gradient_ms = measure(model.gradient)
        try:
            potential_ms = measure(model.hamiltonian)
        except (RuntimeError, ValueError, NotImplementedError, AttributeError):
            potential_ms = float('nan')
    finally:
        model.train(was_training)
    return {'hamiltonian_potential_forward_ms': potential_ms,
            'hamiltonian_gradient_ms': gradient_ms,
            'hamiltonian_total_block_ms': gradient_ms,
            'hamiltonian_probe_repetitions': repetitions,
            'hamiltonian_probe_warmup': warmup,
            'hamiltonian_probe_batch_size': 1}
