from typing import List
import time
from numpy.core.numeric import ones
import pybullet as pb
import numpy as np
from predicates import AtomColor, AtomObject

from sampling import ObjectOnTable, get_on_table_relation, get_near_relation, get_on_relation
from environment import init_env, reset_env, create_shape
from utils import vec3

threshold = 0.3
max_bounds = vec3(7.0, 5.0, 0.0)


def main():
    objects = create_scene()

    while pb.isConnected():
        pb.stepSimulation()
        time.sleep(1.0 / 240.0)

        key_events = pb.getKeyboardEvents()

        # triggered on SPACE key
        if 32 in key_events and key_events[32] & pb.KEY_WAS_RELEASED:
            annotate_scene(objects)


def create_scene():
    objects = [
        ObjectOnTable(AtomObject.CUBE, AtomColor.RED, 2, vec3(-5, -1, 1), vec3(0, 0, 1)),
        ObjectOnTable(AtomObject.CUBE, AtomColor.YELLOW, 2, vec3(-5, -1, 3), vec3(0, 0, -1)),
        ObjectOnTable(AtomObject.PYRAMID, AtomColor.BLUE, 2, vec3(-5, -1, 5), vec3(0, 0, 0)),
        ObjectOnTable(AtomObject.CUBE, AtomColor.GREEN, 2, vec3(0, -1, 1), vec3(0, 0, 2)),
    ]

    init_env(load_egl=False, mode=pb.GUI)
    reset_env()
    for obj in objects:
        obj.shape_id = create_shape(obj.obj_type, obj.color, obj.position, obj.orientation)

    return objects


def annotate_scene(objects):
    relations = []

    for obj in objects:
        _update_position(obj)
        relations += get_on_table_relation(obj, max_bounds)

    relations += get_near_relation(objects, max_distance=4)
    relations += get_on_relation(objects)

    print("---")
    for r in relations:
        if r[1] > threshold:
            print(f"{r[0]}: {r[1]:.3f}")


def _update_position(obj):
    pos, q_orient = pb.getBasePositionAndOrientation(obj.shape_id)
    obj.position = np.array(pos)
    obj.orientation = np.array(pb.getEulerFromQuaternion(q_orient))


if __name__ == "__main__":
    main()
