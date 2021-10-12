import numpy as np
from dataclasses import dataclass

from sampling import sample_one
from environment import grab_frame


class EasyBaseline:
    frame_size = (64, 64)

    def sample():
        predicates = sample_one(
            origin=(0, 0, 1), max_bounds=(0, 0, 0), max_rotation=np.pi * 2.0, num_false_predicates=1
        )
        frame = grab_frame(
            cam_pos=(0, -5, 1),
            cam_target=(0, 0, 1),
            light_dir=(-6, 1, 10),
            frame_size=EasyBaseline.frame_size,
        )
        return frame, predicates


class OneObjectRandomPosition:
    frame_size = (128, 128)

    def sample():
        predicates = sample_one(
            origin=(0, 0, 1.0), max_bounds=(4.0, 4.0, 0.0), max_rotation=2 * np.pi, num_false_predicates=1
        )
        frame = grab_frame(
            cam_pos=(0, -10, 12),
            cam_target=(0, -1, 0),
            light_dir=(-6, 1, 10),
            frame_size=OneObjectRandomPosition.frame_size,
        )
        return frame, predicates
