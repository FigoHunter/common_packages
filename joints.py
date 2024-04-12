import numpy as np
import copy
import torch


BODY_JOINT_NAMES = [
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]
BODY_JOINT_PARENT={
    'left_hip':'root',
    'right_hip': 'root',
    'spine1':'root',
    'left_knee':'left_hip',
    'right_knee':'right_hip',
    'spine2':'spine1',
    'left_ankle':'left_knee',
    'right_ankle':'right_knee',
    'spine3':'spine2',
    'left_foot':'left_ankle',
    'right_foot':'right_ankle',
    'neck':'spine3',
    'left_collar':'neck',
    'right_collar':'neck',
    'head':'neck',
    'left_shoulder':'left_collar',
    'right_shoulder':'right_collar',
    'left_elbow':'left_shoulder',
    'right_elbow':'right_shoulder',
    'left_wrist':'left_elbow',
    'right_wrist':'right_elbow',
}
FINGER_JOINT_NAMES = [
    'index1',
    'index2',
    'index3',
    'middle1',
    'middle2',
    'middle3',
    'pinky1',
    'pinky2',
    'pinky3',
    'ring1',
    'ring2',
    'ring3',
    'thumb1',
    'thumb2',
    'thumb3',
]
FINGER_JOINT_PARENT={
    'index1':'root',
    'index2':'index1',
    'index3':'index2',
    'middle1':'root',
    'middle2':'middle1',
    'middle3':'middle2',
    'pinky1':'root',
    'pinky2':'pinky1',
    'pinky3':'pinky2',
    'ring1':'root',
    'ring2':'ring1',
    'ring3':'ring2',
    'thumb1':'root',
    'thumb2':'thumb1',
    'thumb3':'thumb2',
}

def mirror_rot(rotvec):
    # 镜像轴角 x 轴，实现轴的左右镜像 [-x,y,z]
    # 再xyz全部取反，实现旋转方向的镜像 [x,-y,-z]

    rotvec=np.array([rotvec[0],-rotvec[1],-rotvec[2]])
    return rotvec


def get_mirrored_joint(name:str):
    assert name.startswith('left') or name.startswith('right')
    if name.startswith('left'):
        return name.replace('left','right',1)
    else:
        return name.replace('right','left',1)

def mirror_full_pose(smplx_pose):
    # 镜像根位置
    global_pos = smplx_pose['transl'][0]
    global_pos[:] = torch.tensor([-global_pos[0],global_pos[1],global_pos[2]])

    # 镜像根旋转
    global_rot = smplx_pose['global_orient'][0]
    global_rot[:] = torch.tensor(mirror_rot(global_rot))
    # 镜像身体姿态
    body_pose=smplx_pose['body_pose'][0]
    mir_body_pose = copy.deepcopy(body_pose)

    for i, name in enumerate(BODY_JOINT_NAMES):
        if name.startswith('left'):
            mir_name = get_mirrored_joint(name)
            mir_idx = BODY_JOINT_NAMES.index(mir_name)
            # 获取对称关节的原始旋转
            mir_rot = body_pose[mir_idx*3:(mir_idx+1)*3]
            # 镜像对称关节旋转
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(mir_rot))
        elif name.startswith('right'):
            mir_name = get_mirrored_joint(name)
            mir_idx = BODY_JOINT_NAMES.index(mir_name)
            mir_rot = body_pose[mir_idx*3:(mir_idx+1)*3]
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(mir_rot))
        else:
            # 镜像非对称关节旋转
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(body_pose[i*3:(i+1)*3]))
    body_pose[:]=mir_body_pose[:]

    # 镜像左右手
    left_hand_pose = smplx_pose['left_hand_pose'][0]
    right_hand_pose = smplx_pose['right_hand_pose'][0].view(-1)
    mir_left_hand_pose = copy.deepcopy(left_hand_pose)
    mir_right_hand_pose = copy.deepcopy(right_hand_pose)

    for i, name in enumerate(FINGER_JOINT_NAMES):
        mir_left_hand_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(right_hand_pose[i*3:(i+1)*3]))
        mir_right_hand_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(left_hand_pose[i*3:(i+1)*3]))

    left_hand_pose[:]=mir_left_hand_pose[:]
    right_hand_pose[:]=mir_right_hand_pose[:]
    smplx_pose['right_hand_pose'][0][:,:]=right_hand_pose.reshape([15,3])[:,:]
