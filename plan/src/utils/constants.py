import numpy as np

IMAGE_WIDTH, IMAGE_HEIGHT = int(256 * 3 / 2), int(256 * 3 / 2)
IMAGE_INTRINSICS = np.array([
    [968 / 2, 0, 128*3/2],
    [0, 968 / 2, 128*3/2],
    [0, 0, 1],
])

CAMERA_ANGLE = 33 * np.pi / 180
IMAGE_EXTRINSICS = np.array([
    [0, np.sin(CAMERA_ANGLE), -np.cos(CAMERA_ANGLE), 1.3],
    [1, 0, 0, 0],
    [0, -np.cos(CAMERA_ANGLE), -np.sin(CAMERA_ANGLE), 0.75],
    [0, 0, 0, 1],
], dtype=np.float32)


# ROBOT_URDF = 'ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/googlerobot_description/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf'
ROBOT_JOINTS_NOFINGER_GOOGLEROBOT = ['joint_torso', 'joint_shoulder', 'joint_bicep', 'joint_elbow', 'joint_forearm', 'joint_wrist', 'joint_gripper', 'joint_head_pan', 'joint_head_tilt'] 
ROBOT_JOINTS_GOOGLEROBOT = ['joint_torso', 'joint_shoulder', 'joint_bicep', 'joint_elbow', 'joint_forearm','joint_wrist', 'joint_gripper', 'joint_finger_right', 'joint_finger_left', 'joint_head_pan', 'joint_head_tilt'] 

# for widowx
# ROBOT_URDF = 'ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/wx250s.urdf'
ROBOT_JOINTS_WIDOWX = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'left_finger', 'right_finger']
ROBOT_JOINTS_NOFINGER_WIDOWX  = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']



# ROBOT_URDF = 'robot_models/franka/franka_with_gripper_extensions.urdf'
# ROBOT_JOINTS = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']


# ARM_URDF = './robot_models/franka/franka_without_gripper.urdf'
ARM_URDF_GOOGLE_ROBOT = 'plan/robot_models/google_robot/google_robot_without_gripper.urdf'
ARM_URDF_FULL_GOOGLE_ROBOT = 'ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/googlerobot_description/ik_google_robot.urdf'
ARM_URDF_WIDOWX = 'plan/robot_models/widowx/widowx_robot_without_gripper.urdf'
ARM_URDF_FULL_WIDOWX = 'ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/scale_wx250s.urdf'
FRANKA_GRIPPER_DEPTH = 0.005
DGN_GRIPPER_DEPTH = 0.04

GRIPPER_HALF_WIDTH = 0.04

MAX_OBJ_NUM = 20

FRANKA_INIT_QPOS = np.array([0.0, 0.22, 0.0, -1, 0.0, 1.3, 0.79, 0.0, 0.0])
FRANKA_NEUTRAL_QPOS = np.array([-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342, 0.0, 0.0])
FRANKA_JOINT_LIMITS = np.array(
        [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (0.5, 3.75),
            (-2.8973, 2.8973),
            (0.0, 0.04),
            (0.0, 0.04),
        ]
    )
RAND_CENTER = np.array([0.5, 0.0])
RAND_RAIDUS = 0.1

# FRANKA_ADDITIONAL_BOXES = dict(
#     panda_leftfinger=[np.array([[-0.04, -0.011     , -0.015     ], [ 8.9941919e-03,  1.1000000e-02, -3.0547379e-10]]),
#                       np.array([[-0.068  , -0.011  , -0.02665], [-0.03000963,  0.011     , -0.00265   ]])],
#     panda_rightfinger=[np.array([[-0.00899431, -0.011     , -0.015     ], [ 0.04,  1.1000000e-02, -3.0547379e-10]]),
#                        np.array([[ 0.03001618, -0.011     , -0.02665   ], [ 0.068  ,  0.011  , -0.00265]])],
#     panda_hand=[np.array([[-0.021, -0.10055307,  0.01500777], [0.021, 0.1005518 , 0.06599596]]),
#                 np.array([[-0.03150471, -0.10394307, -0.02591177], [0.03150447, 0.09628979, 0.01499266]])],
# )

FRANKA_COLLISION_FILE = 'plan/robot_models/franka/curobo/franka_mesh.yml'
# FRANKA_CUROBO_FILE = 'robot_models/franka/curobo/franka.yml'
FRANKA_CUROBO_FILE = 'plan/robot_models/google_robot/curobo/google_robot.yml'
