import pkgutil

import pybullet as pb
import pybullet_data


def init_env(load_egl=True):
    client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    if load_egl:
        egl = pkgutil.get_loader("eglRenderer")
        pb.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

    return client


def reset_env():
    pb.resetSimulation()
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    pb.setGravity(0, 0, -9.8)

    plane = pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.5], useFixedBase=True, globalScaling=2)
    table = pb.loadURDF("table/table.urdf", basePosition=[0, 0, -6.3], useFixedBase=True, globalScaling=10)


def disconnect_env():
    pb.disconnect()


def get_camera_transforms(position, target):
    projectionMatrix = pb.computeProjectionMatrixFOV(fov=45.0, aspect=1.0, nearVal=0.1, farVal=100.0)

    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=position, cameraTargetPosition=target, cameraUpVector=[0, 0, 1]
    )
    return viewMatrix, projectionMatrix


def create_shape(
    shape_type, mass, position, half_extends=0, radius=0, mesh_scale=None, orientation=None, color=None
):
    if not orientation:
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


def grab_frame(cam_pos, cam_target, light_dir, frame_size):
    cam_view_m, cam_proj_m = get_camera_transforms(cam_pos, cam_target)
    _, _, rgbImg, _, _ = pb.getCameraImage(*frame_size, cam_view_m, cam_proj_m, lightDirection=light_dir)

    return rgbImg[..., :3]
