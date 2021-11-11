import pkgutil

import pybullet as pb
import pybullet_data

from predicates import AtomColor, AtomObject


def init_env(load_egl=True, mode=pb.DIRECT):
    client = pb.connect(mode)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    if load_egl:
        egl = pkgutil.get_loader("eglRenderer")
        pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

    return client


def reset_env():
    pb.resetSimulation()
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    pb.setGravity(0, 0, -9.8)

    pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.5], useFixedBase=True, globalScaling=2)
    pb.loadURDF("table/table.urdf", basePosition=[0, 0, -6.3], useFixedBase=True, globalScaling=10)


def disconnect_env():
    pb.disconnect()


def get_camera_transforms(position, target):
    projectionMatrix = pb.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=100.0)

    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=position, cameraTargetPosition=target, cameraUpVector=[0, 0, 1]
    )
    return viewMatrix, projectionMatrix


def grab_frame(cam_pos, cam_target, light_dir, frame_size):
    cam_view_m, cam_proj_m = get_camera_transforms(cam_pos, cam_target)
    _, _, rgbImg, _, _ = pb.getCameraImage(*frame_size, cam_view_m, cam_proj_m, lightDirection=light_dir)

    return rgbImg[..., :3]


def _create_shape(
    shape_type, mass, position, half_extends=0, radius=0, mesh_scale=None, orientation=None, color=None
):
    if orientation is None:
        orientation = [0, 0, 0]

    if isinstance(shape_type, str):
        if not mesh_scale:
            mesh_scale = [1, 1, 1]
        vId = pb.createVisualShape(
            shapeType=pb.GEOM_MESH, fileName=shape_type, meshScale=mesh_scale, rgbaColor=color
        )
        cId = pb.createCollisionShape(shapeType=pb.GEOM_MESH, fileName=shape_type, meshScale=mesh_scale)
    else:
        vId = pb.createVisualShape(
            shapeType=shape_type, halfExtents=half_extends, radius=radius, rgbaColor=color
        )
        cId = pb.createCollisionShape(shapeType=shape_type, halfExtents=half_extends, radius=radius)
    return pb.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=cId,
        baseVisualShapeIndex=vId,
        basePosition=position,
        baseOrientation=pb.getQuaternionFromEuler(orientation),
    )


def create_shape(obj: AtomObject, color: AtomColor, position, orientation, size=2.0):
    if obj == AtomObject.CUBE:
        return _create_shape(
            pb.GEOM_BOX,
            1,
            position,
            orientation=orientation,
            half_extends=[size * 0.5, size * 0.5, size * 0.5],
            color=color.to_rgba(),
        )
    elif obj == AtomObject.SPHERE:
        return _create_shape(
            pb.GEOM_SPHERE, 1, position, orientation=orientation, radius=size * 0.5, color=color.to_rgba()
        )
    elif obj == AtomObject.PYRAMID:
        return _create_shape(
            "shapes/pyramid.obj",
            1,
            position,
            orientation=orientation,
            mesh_scale=[size, size, size],
            color=color.to_rgba(),
        )
    else:
        raise ValueError("Unsupported shape!")
