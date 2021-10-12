from typing import Sequence
import random

import numpy as np
from tensorflow import keras

from sampling import sample_one
from predicates import AtomPredicate
from environment import init_env, disconnect_env
from utils import HideOutput


def _create_samples_old(num_samples, num_false_predicates, frame_size):
    origin = (0, 0, 0.5)
    bounds = (4.0, 4.0, 0.0)

    with HideOutput():
        init_env()

        raw_samples = [
            (frame, p.to_one_hot(), np.array([p_value]))
            for frame, predicates in [
                sample_one(origin, bounds, num_false_predicates, 0, frame_size) for i in range(num_samples)
            ]
            for p, p_value in predicates
        ]
        disconnect_env()

    random.shuffle(raw_samples)
    return raw_samples


class OneObjectGenerator(keras.utils.Sequence):
    def __init__(self, num_samples, num_false_predicates, frame_size, batch_size=32, verbose=True):
        self.num_samples = num_samples
        self.frame_size = frame_size
        self.num_false_p = num_false_predicates
        self.samples = None
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_batches = -1
        self.on_epoch_end()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        return self.samples[index]

    def _to_tensor(predicates: Sequence[AtomPredicate]):
        one_hot, p_values = zip(*[(p.to_one_hot(), np.array([p_value])) for p, p_value in predicates])
        return np.stack(one_hot, axis=0), np.stack(p_values, axis=0)

    def on_epoch_end(self):
        if self.verbose:
            print("Generating epoch data...", end="")

        raw_samples = _create_samples_old(self.num_samples, self.num_false_p, self.frame_size)
        self.num_batches = len(raw_samples) // self.batch_size

        batches = [
            raw_samples[i * self.batch_size : (i + 1) * self.batch_size] for i in range(self.num_batches)
        ]
        batches = [tuple(zip(*batch)) for batch in batches]

        self.samples = [
            ((np.stack(batch[0], axis=0), np.stack(batch[1], axis=0)), np.stack(batch[2], axis=0))
            for batch in batches
        ]

        if self.verbose:
            print("OK")
