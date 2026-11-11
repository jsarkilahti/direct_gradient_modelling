"""Select representative runs for comparison figures and evaluation summaries."""
from collections import defaultdict
import math


def select_representatives(study, *, test_metric='nrmse'):
    """Select representative runs for comparison and prediction plots."""
    groups = defaultdict(list)
    for index, run in enumerate(study.runs):
        groups[(run.configuration.get('level'), run.method)].append((index, run))
    selected = {}
    for (level, method), entries in groups.items():
        def validation(item):
            """Handle validation for the experiment workflow."""
            _, run = item
            value = run.metrics.get('best_validation_loss')
            if value is None:
                value = min(run.validation_loss, default=float('nan'))
            return float(value)
        def test(item):
            """Handle test for the experiment workflow."""
            return float(item[1].metrics.get(test_metric, float('nan')))
        def choose(score):
            """Handle choose for the experiment workflow."""
            valid = [(score(item), item) for item in entries]
            valid = [(v, item) for v, item in valid if math.isfinite(v)]
            if not valid:
                return None
            value, (index, run) = min(valid, key=lambda p: (p[0], p[1][0]))
            return {'run_index': index, 'model_seed': run.model_seed,
                    'shuffle_seed': run.shuffle_seed, 'score': value}
        selected[f'{method}|{level}'] = {
            'validation_selected': choose(validation),
            'test_selected_descriptive_only': choose(test),
            'test_metric': test_metric,
        }
    return selected
