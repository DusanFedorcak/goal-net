import numpy as np

from environment import init_env, disconnect_env
from predicates import AtomColor, AtomObject, AtomRelation, AtomPredicate
from sampling import _VALID_COLORS, _VALID_SHAPES
from utils import draw_frame, HideOutput


def evaluate_sample(sample_func, model=None, probes=None, threshold=0.3, hide_output=True):

    with HideOutput(hide_output):
        init_env()
        test_frame, truth = sample_func()
        disconnect_env()

    draw_frame(test_frame)
    for p, v in truth:
        print(p, v)

    if model:
        prediction = model.predict((np.broadcast_to(test_frame, (len(probes), *test_frame.shape)), probes,))

        print("---")
        for probe, pred in sorted(zip(probes, prediction), key=lambda x: -x[1]):
            if pred[0] > threshold:
                print(AtomPredicate.from_one_hot(probe), pred[0])

