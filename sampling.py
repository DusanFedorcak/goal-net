from typing import Sequence, Tuple, Union, List
import dataclasses
from dataclasses import dataclass
from itertools import permutations

import random

import numpy as np
import pybullet as pb

from environment import reset_env, create_shape as create_bullet_shape
from predicates import AtomColor, AtomObject, AtomPredicate, AtomRelation


def _sigmoid(x):
    return 1 / (1 + np.exp(-10 * x))


def _act(x):
    return np.clip(x, 0, 1)


_POSITION_TESTS = {
    AtomRelation.ON_LEFT_SIDE_OF: lambda x: _act(-x[0]),
    AtomRelation.ON_RIGHT_SIDE_OF: lambda x: _act(x[0]),
    AtomRelation.ON_FAR_SIDE_OF: lambda x: _act(x[1]),
    AtomRelation.ON_NEAR_SIDE_OF: lambda x: _act(-x[1]),
    AtomRelation.IN_CENTER_OF: lambda x: _act(1.0 - np.linalg.norm(x)),
}

_VALID_SHAPES = {AtomObject.CUBE, AtomObject.SPHERE, AtomObject.PYRAMID}
_VALID_COLORS = set(AtomColor.__members__.values()) - {AtomColor.NO_COLOR, AtomColor.WHITE}


@dataclass
class ObjectOnTable:
    obj_type: AtomObject
    color: AtomColor
    position: np.array
    orientation: np.array

    def sample(origin: np.array, max_bounds: Union[np.array, float], max_rotation: float) -> "ObjectOnTable":
        return ObjectOnTable(
            random.sample(_VALID_SHAPES, 1)[0],
            random.sample(_VALID_COLORS, 1)[0],
            (np.random.random_sample(3) * 2 - 1) * max_bounds + origin,
            [0, 0, np.random.ranf() * max_rotation],
        )

    def _has_intersection(o1: "ObjectOnTable", o2: "ObjectOnTable", min_distance: float) -> bool:
        return np.linalg.norm(o1.position - o2.position) < min_distance

    def sample_no_intersection(
        origin: np.array,
        max_bounds: Union[np.array, float],
        max_rotation: float,
        existing_objects: Sequence["ObjectOnTable"],
        min_distance: float,
    ) -> "ObjectOnTable":
        sample = ObjectOnTable.sample(origin, max_bounds, max_rotation)
        while any(ObjectOnTable._has_intersection(sample, o, min_distance) for o in existing_objects):
            sample = ObjectOnTable.sample(origin, max_bounds, max_rotation)

        return sample


def _test_position(
    position: np.array, max_bounds: Union[np.array, float]
) -> List[Tuple[AtomRelation, float]]:
    max_bounds = np.array(max_bounds)
    fixed_bounds = max_bounds[max_bounds == 0] = 1.0
    norm_p = position / fixed_bounds
    return [(relation, float(test(norm_p))) for relation, test in _POSITION_TESTS.items()]


def get_on_table_relation(o: ObjectOnTable, max_bounds: Union[np.array, float], add_positional=True):
    preds = [(AtomPredicate(AtomRelation.ON, o.obj_type, o.color, AtomObject.TABLE, AtomColor.NO_COLOR), 1.0)]

    if add_positional:
        preds += [
            (AtomPredicate(relation, o.obj_type, o.color, AtomObject.TABLE, AtomColor.NO_COLOR), value)
            for relation, value in _test_position(o.position[:2], max_bounds[:2])
        ]

    return preds


def get_near_relation(objects: Sequence[ObjectOnTable], max_distance=4.0):
    return [
        (
            AtomPredicate(AtomRelation.NEAR, o1.obj_type, o1.color, o2.obj_type, o2.color),
            np.clip(2.0 * (1.0 - np.linalg.norm(o1.position - o2.position) / max_distance), 0, 1),
        )
        for o1, o2 in permutations(objects, 2)
    ]


def _obj_complement(predicate: AtomPredicate):
    return [
        dataclasses.replace(predicate, obj=o, obj_color=oc)
        for o in _VALID_SHAPES
        for oc in _VALID_COLORS
        if o != predicate.obj or oc != predicate.obj_color
    ]


def get_false_predicates(predicates, count):
    p_set = set(next(zip(*predicates)))
    false_predicates = [
        (false_p, 0.0) for p, _ in predicates for false_p in _obj_complement(p) if false_p not in p_set
    ]
    return random.sample(false_predicates, count) if count < len(false_predicates) else false_predicates


def create_shape(o: ObjectOnTable):
    return _create_shape(o.obj_type, o.color, o.position, o.orientation)


def _create_shape(obj: AtomObject, color: AtomColor, position, orientation):
    if obj == AtomObject.CUBE:
        return create_bullet_shape(
            pb.GEOM_BOX, 1, position, orientation=orientation, half_extends=[1, 1, 1], color=color.to_rgba(),
        )
    elif obj == AtomObject.SPHERE:
        return create_bullet_shape(
            pb.GEOM_SPHERE, 1, position, orientation=orientation, radius=1, color=color.to_rgba()
        )
    elif obj == AtomObject.PYRAMID:
        return create_bullet_shape(
            "pyramid.obj", 1, position, orientation=orientation, mesh_scale=[2, 2, 2], color=color.to_rgba(),
        )
    else:
        raise ValueError("Unsupported shape!")


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


def __sample_one(
    origin: np.array, max_bounds: Union[np.array, float], max_rotation: float, num_false_predicates: int,
):
    # sample attributes
    obj = random.sample(_VALID_SHAPES, 1)[0]
    color = random.sample(_VALID_COLORS, 1)[0]
    position = (np.random.random_sample(3) * 2 - 1) * max_bounds + origin
    orientation = [0, 0, np.random.ranf() * max_rotation]

    # construct predicates
    on_predicates = [(AtomPredicate(AtomRelation.ON, obj, color, AtomObject.TABLE, AtomColor.NO_COLOR), 1.0)]

    position_predicates = [
        (AtomPredicate(relation, obj, color, AtomObject.TABLE, AtomColor.NO_COLOR), value)
        for relation, value in _test_position(position[:2], max_bounds[:2])
    ]

    predicates = on_predicates + position_predicates
    false_predicates = [(false_p, 0.0) for p, _ in predicates for false_p in _obj_complement(p)]

    # add shape to scene
    reset_env()
    _create_shape(obj, color, position, orientation)

    return predicates + (
        random.sample(false_predicates, num_false_predicates)
        if num_false_predicates < len(false_predicates)
        else false_predicates
    )
