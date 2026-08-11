import json
import random

import numpy as np
import torch

from .config import TRAIN_CONFIG


REPORT_METRICS = [
    "step_micro_accuracy",
    "agent_micro_accuracy",
]


def load_samples(paths):
    samples = []
    for path in paths:
        samples.extend(torch.load(path, map_location="cpu", weights_only=False))
    return samples


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data(samples, seed):
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    count = max(1, round(len(indices) * TRAIN_CONFIG["validation_ratio"]))
    validation_ids = set(indices[:count])
    train = [sample for index, sample in enumerate(samples) if index not in validation_ids]
    validation = [sample for index, sample in enumerate(samples) if index in validation_ids]
    return train, validation


def positive_weight(samples, key, device):
    positive = sum(int((sample[key] > 0.5).sum()) for sample in samples)
    total = sum(sample[key].numel() for sample in samples)
    return torch.tensor((total - positive) / max(positive, 1), device=device)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
