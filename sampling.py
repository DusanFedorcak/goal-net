from typing import Dict, Sequence, Tuple, Union, List, Optional
import dataclasses
from dataclasses import dataclass
from itertools import permutations
import random

import numpy as np

from predicates import AtomColor, AtomObject, AtomPredicate, AtomRelation


@dataclass
class ObjectOnTable:
    obj_type: AtomObject
    color: AtomColor
    size: float
    position: np.array
    orientation: np.array
    shape_id: Optional[int] = None

    def sample(
        origin: np.array, max_bounds: Union[np.array, float], max_rotation: float, size: float
    ) -> "ObjectOnTable":
        return ObjectOnTable(
            random.sample(_VALID_SHAPES, 1)[0],
            random.sample(_VALID_COLORS, 1)[0],
            size,
            (np.random.random_sample(3) * 2 - 1) * max_bounds + origin,
            np.array([0, 0, np.random.ranf() * max_rotation]),
        )

    def _has_intersection(o1: "ObjectOnTable", o2: "ObjectOnTable", min_distance: float) -> bool:
        return np.linalg.norm(o1.position - o2.position) < min_distance

    def sample_no_intersection(
        origin: np.array,
        max_bounds: Union[np.array, float],
        max_rotation: float,
        size: float,
        existing_objects: Sequence["ObjectOnTable"],
        min_distance: float,
    ) -> "ObjectOnTable":
        sample = ObjectOnTable.sample(origin, max_bounds, max_rotation, size)
        while any(ObjectOnTable._has_intersection(sample, o, min_distance) for o in existing_objects):
            sample = ObjectOnTable.sample(origin, max_bounds, max_rotation, size)

        return sample


_VALID_SHAPES = {AtomObject.CUBE, AtomObject.SPHERE, AtomObject.PYRAMID}
_VALID_COLORS = set(AtomColor.__members__.values()) - {AtomColor.NO_COLOR, AtomColor.WHITE}


def get_on_table_relation(
    obj: ObjectOnTable, max_bounds: Union[np.array, float], add_positional=True
) -> List[Tuple[AtomPredicate, float]]:
    # normalize obj position against bounds
    max_bounds = np.array(max_bounds)
    max_bounds[max_bounds == 0] = 1.0
    norm_pos = obj.position / max_bounds
    # test table positions
    tests = {relation: float(test(obj, norm_pos)) for relation, test in _TABLE_POSITION_TESTS.items()}

    is_on_table = tests[AtomRelation.ON]
    preds = [
        (
            AtomPredicate(AtomRelation.ON, obj.obj_type, obj.color, AtomObject.TABLE, AtomColor.NO_COLOR),
            is_on_table,
        ),
    ]

    if add_positional and is_on_table:
        preds += [
            (AtomPredicate(relation, obj.obj_type, obj.color, AtomObject.TABLE, AtomColor.NO_COLOR), value)
            for relation, value in tests.items()
            if relation != AtomRelation.ON
        ]

    return preds


_EPS = 0.05
_TABLE_POSITION_TESTS = {
    AtomRelation.ON: lambda obj, norm_pos: float(
        all(np.abs(norm_pos[:2]) < 1.0) and obj.position[2] < obj.size * 0.5 + _EPS
    ),
    AtomRelation.ON_LEFT_SIDE_OF: lambda obj, norm_pos: _clip(-norm_pos[0]),
    AtomRelation.ON_RIGHT_SIDE_OF: lambda obj, norm_pos: _clip(norm_pos[0]),
    AtomRelation.ON_FAR_SIDE_OF: lambda obj, norm_pos: _clip(norm_pos[1]),
    AtomRelation.ON_NEAR_SIDE_OF: lambda obj, norm_pos: _clip(-norm_pos[1]),
    AtomRelation.IN_CENTER_OF: lambda obj, norm_pos: _clip(1.0 - 1.5 * np.linalg.norm(norm_pos[:2])),
}


def _sigm(x, weight=1.0):
    return 1 / (1 + np.exp(-weight * x))


def _clip(x):
    return np.clip(x, 0, 1)


def get_near_relation(
    objects: Sequence[ObjectOnTable], max_distance=4.0
) -> List[Tuple[AtomPredicate, float]]:
    return [
        (
            AtomPredicate(AtomRelation.NEAR, o1.obj_type, o1.color, o2.obj_type, o2.color),
            np.clip(2.0 * (1.0 - np.linalg.norm(o1.position - o2.position) / max_distance), 0, 1),
        )
        for o1, o2 in permutations(objects, 2)
    ]


def get_on_relation(objects: Sequence[ObjectOnTable]):
    return [
        (AtomPredicate(AtomRelation.ON, o1.obj_type, o1.color, o2.obj_type, o2.color), 1.0)
        for o1, o2 in permutations(objects, 2)
        if (
            np.linalg.norm(o1.position[:2] - o2.position[:2]) - 0.5 * (o1.size + o2.size) < 0
            and np.abs(o1.position[2] - (o2.position[2] + 0.5 * (o1.size + o2.size))) < o1.size * 0.5
        )
    ]


def _obj_complement(predicate: AtomPredicate) -> List[ObjectOnTable]:
    return [
        dataclasses.replace(predicate, obj=o, obj_color=oc)
        for o in _VALID_SHAPES
        for oc in _VALID_COLORS
        if o != predicate.obj or oc != predicate.obj_color
    ]


def get_false_predicates(
    predicates: Sequence[AtomPredicate], count: int
) -> List[Tuple[AtomPredicate, float]]:
    p_set = set(next(zip(*predicates)))
    false_predicates = [
        (false_p, 0.0) for p, _ in predicates for false_p in _obj_complement(p) if false_p not in p_set
    ]
    return random.sample(false_predicates, count) if count < len(false_predicates) else false_predicates


def get_on_table_probes(add_positional=True):
    relations = (
        set(AtomRelation.__members__.values()) - {AtomRelation.NEAR} if add_positional else {AtomRelation.ON}
    )

    return np.vstack(
        [
            AtomPredicate(relation, obj, obj_color, AtomObject.TABLE, AtomColor.NO_COLOR).to_one_hot()
            for relation in relations
            for obj in _VALID_SHAPES
            for obj_color in _VALID_COLORS
        ]
    )


def get_near_probes():
    return np.vstack(
        [
            AtomPredicate(AtomRelation.NEAR, obj_1, obj_1_color, obj_2, obj_2_color).to_one_hot()
            for obj_1 in _VALID_SHAPES
            for obj_1_color in _VALID_COLORS
            for obj_2 in _VALID_SHAPES
            for obj_2_color in _VALID_COLORS
        ]
    )
