import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm


def _to_tensor(predicates):
    indices = np.random.permutation(range(len(predicates)))
    preds, p_values = zip(*[(p.to_one_hot(), np.array([p_value])) for p, p_value in predicates])
    return np.stack(preds, axis=0)[indices], np.stack(p_values, axis=0)[indices]


def _create_samples(num_samples, sample_func, hide_output=True):
    from environment import init_env, disconnect_env
    from utils import HideOutput

    with HideOutput(hide_output):
        init_env()
        raw_samples = [
            (frame, *_to_tensor(predicates))
            for frame, predicates in [sample_func() for i in range(num_samples)]
        ]
        disconnect_env()

    return raw_samples


def create_dataset(path: str, num_batches: int, batch_size: int, sample_func, hide_output=True):
    print(f"Generating {num_batches} * {batch_size} = {num_batches * batch_size} samples")

    _order = int(np.log10(num_batches)) + 1
    os.makedirs(path)

    for i in tqdm(range(num_batches)):
        batch = _create_samples(batch_size, sample_func, hide_output)
        frames, predicates, targets = zip(*batch)
        np.savez(
            path + f"/_{str(i).zfill(_order)}.npz",
            frames=np.stack(frames, axis=0),
            predicates=np.stack(predicates, axis=0),
            targets=np.stack(targets, axis=0),
        )


def _parse_files(f):
    data = np.load(f.numpy())
    frames = data["frames"]
    predicates = data["predicates"]
    targets = data["targets"]
    return frames, predicates, targets


def _process(f):
    frames, predicates, targets = tf.py_function(_parse_files, [f], [tf.float32, tf.float32, tf.float32])
    return tf.data.Dataset.from_tensor_slices(
        (
            (
                tf.repeat(frames, tf.shape(targets)[1], axis=0),
                tf.reshape(predicates, (-1, tf.shape(predicates)[-1])),
            ),
            tf.reshape(targets, (-1, tf.shape(targets)[-1])),
        )
    )


def load_as_tf_dataset(*path):
    return tf.data.Dataset.list_files(path + "/*.npz").flat_map(_process)


def _list_files(path):
    return tf.data.Dataset.list_files(path + "/*.npz")


def load_datasets(paths):
    files = tf.data.Dataset.from_tensors(paths).flat_map(_list_files)
    num_files = len(list(files))
    print(f"Datasets contain {num_files} files in total.")
    files = files.shuffle(num_files)

    return files.flat_map(_process)

