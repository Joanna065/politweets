import numpy as np
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import Sampler, WeightedRandomSampler

_SAMPLERS = [
    'weighted_sampler',
    'm_per_class_sampler',
]


def get_sampler(sampler_name: str, labels: np.ndarray, batch_size: int) -> Sampler:
    if sampler_name not in _SAMPLERS:
        raise ValueError("Unsupported sampler name.")

    if sampler_name == 'weighted_sampler':
        labels_sample_count = np.unique(labels, return_counts=True)[1]
        weights = 1. / labels_sample_count
        samples_weights = weights[labels]

        return WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
        )

    if sampler_name == 'm_per_class_sampler':
        num_labels = len(np.unique(labels))
        return MPerClassSampler(
            labels=labels,
            m=batch_size // num_labels,
            batch_size=batch_size,
            length_before_new_iter=len(labels),
        )
