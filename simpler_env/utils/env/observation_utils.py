import numpy as np
from transforms3d.quaternions import quat2mat
def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]

def get_depth_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain depth from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["depth"]

def get_pointcloud_in_camera(env, obs, camera_name):
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
    camera_param = obs["camera_param"][camera_name]
    depth = obs["image"][camera_name]["depth"]
    intrinsic_cv = camera_param["intrinsic_cv"]
    extrinsic_cv = camera_param["extrinsic_cv"]
    # cam2world_gl = camera_param['cam2world_gl']
    h, w = depth.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = depth.flatten()
    xx = (xx - intrinsic_cv[0, 2]) / intrinsic_cv[0, 0] * zz
    yy = (yy - intrinsic_cv[1, 2]) / intrinsic_cv[1, 1] * zz
    ones = np.ones_like(zz)
    points = np.stack([xx, yy, zz, ones], axis=0)  # [3, N]
    points_world = invert_transform(extrinsic_cv) @ points
    # Convert to base frame
    base_pose = obs["agent"]["base_pose"]  # [p, q]
    base_T = np.eye(4)
    base_T[:3, :3] = quat2mat(base_pose[3:])
    base_T[:3, 3] = base_pose[:3]
    points_base = invert_transform(base_T) @ points_world
    points_rgb = obs["image"][camera_name]["rgb"].reshape(-1, 3)
    
    # import matplotlib.pyplot as plt
    # plt.imshow(obs['image'][camera_name]['rgb'])
    # plt.show()
    return points[:3].T, points_base[:3].T, points_rgb


def invert_transform(H: np.ndarray):
    """
    Compute the inverse of a 4x4 transformation matrix.

    This function is a simple wrapper around the standard matrix inversion
    formula for 4x4 transformation matrices. It is written to be vectorized,
    so it can be applied to an array of transformation matrices.

    Parameters
    ----------
    H: array-like
        The transformation matrix to invert. Should have shape (4, 4) or
        (..., 4, 4).

    Returns
    -------
    H_inv: array-like
        The inverse of the transformation matrix. Has the same shape as H.
    """
    assert H.shape[-2:] == (4, 4), H.shape
    H_inv = H.copy()
    R_T = np.swapaxes(H[..., :3, :3], -1, -2)
    H_inv[..., :3, :3] = R_T
    H_inv[..., :3, 3:] = -R_T @ H[..., :3, 3:]
    return H_inv

def get_camera_extrinsics_from_maniskill2_obs_dict(env, obs, camera_name): 
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    camera_param = obs["camera_param"][camera_name]
    intrinsic_cv = camera_param["intrinsic_cv"]
    extrinsic_cv = camera_param["extrinsic_cv"]
    return intrinsic_cv, extrinsic_cv


def get_base_pose(obs):
    base_pose = obs["agent"]["base_pose"]  # [p, q]
    base_T = np.eye(4)
    base_T[:3, :3] = quat2mat(base_pose[3:])
    base_T[:3, 3] = base_pose[:3]
    return base_T