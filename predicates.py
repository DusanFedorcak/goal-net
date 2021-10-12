import dataclasses
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class AtomColor(IntEnum):
    NO_COLOR = 0
    BLUE = 1
    GREEN = 2
    CYAN = 3
    RED = 4
    PURPLE = 5
    YELLOW = 6
    WHITE = 7

    def nice_name(self):
        return self.name.lower() if self else ""

    def to_rgba(self):
        return tuple(float(ch) for ch in np.binary_repr(self, width=3)) + tuple([1.0])


class AtomObject(IntEnum):
    ACTOR = 0
    TABLE = 1
    CUBE = 2
    SPHERE = 3
    PYRAMID = 4

    def nice_name(self):
        return self.name.lower()


class AtomRelation(IntEnum):
    ON = 0
    NEAR = 1
    ON_LEFT_SIDE_OF = 2
    ON_RIGHT_SIDE_OF = 3
    ON_NEAR_SIDE_OF = 4
    ON_FAR_SIDE_OF = 5
    IN_CENTER_OF = 6

    def nice_name(self):
        return self.name.lower()


def to_one_hot(e: IntEnum):
    return np.eye(len(e.__class__.__members__))[e]


def from_one_hot(arr: np.array):
    return np.nonzero(arr)[0][0]


@dataclass(frozen=True)
class AtomPredicate:
    relation: AtomRelation
    obj: AtomObject
    obj_color: AtomColor
    subj: AtomObject
    subj_color: AtomColor

    def __repr__(self):
        return f"""({self.obj_color.nice_name()} {self.obj.nice_name()} {self.relation.nice_name()} {self.subj_color.nice_name()} {self.subj.nice_name()})""".replace(
            "  ", " "
        )

    def to_one_hot(self):
        return np.concatenate([to_one_hot(e) for e in dataclasses.astuple(self)])

    def from_one_hot(arr: np.array):
        fields = [f.type for f in dataclasses.fields(AtomPredicate)]
        indices = np.cumsum([len(e.__members__) for e in fields])
        return AtomPredicate(*(E(from_one_hot(arr)) for arr, E in zip(np.split(arr, indices), fields)))
