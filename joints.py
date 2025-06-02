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

SMPL_BODY_JOINT_NAMES = [
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
    'left_hand',
    'right_hand',
]
SMPL_BODY_JOINT_PARENT={
    'left_hip':'root',
    'right_hip':'root',
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
    'left_collar':'spine3',
    'right_collar':'spine3',
    'head':'neck',
    'left_shoulder':'left_collar',
    'right_shoulder':'right_collar',
    'left_elbow':'left_shoulder',
    'right_elbow':'right_shoulder',
    'left_wrist':'left_elbow',
    'right_wrist':'right_elbow',
    'left_hand':'left_wrist',
    'right_hand':'right_wrist',
}


def get_joint_idx(joint_name):
    if joint_name in BODY_JOINT_NAMES:
        return BODY_JOINT_NAMES.index(joint_name)
    elif joint_name in FINGER_JOINT_NAMES:
        return FINGER_JOINT_NAMES.index(joint_name)
    elif joint_name == 'root':
        return -1
    else:
        raise ValueError('Unknown joint name: {}'.format(joint_name))

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

def get_smpl_joint_idx(joint_name):
    if joint_name in SMPL_BODY_JOINT_NAMES:
        return SMPL_BODY_JOINT_NAMES.index(joint_name)
    elif joint_name == 'root':
        return -1
    else:
        raise ValueError('Unknown joint name: {}'.format(joint_name))
    
def mirror_smpl_pose(smpl_pose):
    # 镜像根位置
    global_pos = smpl_pose['transl'][0]
    global_pos[:] = torch.tensor([-global_pos[0],global_pos[1],global_pos[2]])

    # 镜像根旋转
    global_rot = smpl_pose['global_orient'][0]
    global_rot[:] = torch.tensor(mirror_rot(global_rot))
    # 镜像身体姿态
    body_pose=smpl_pose['body_pose'][0]
    mir_body_pose = copy.deepcopy(body_pose)

    for i, name in enumerate(SMPL_BODY_JOINT_NAMES):
        if name.startswith('left'):
            mir_name = get_mirrored_joint(name)
            mir_idx = SMPL_BODY_JOINT_NAMES.index(mir_name)
            # 获取对称关节的原始旋转
            mir_rot = body_pose[mir_idx*3:(mir_idx+1)*3]
            # 镜像对称关节旋转
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(mir_rot))
        elif name.startswith('right'):
            mir_name = get_mirrored_joint(name)
            mir_idx = SMPL_BODY_JOINT_NAMES.index(mir_name)
            mir_rot = body_pose[mir_idx*3:(mir_idx+1)*3]
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(mir_rot))
        else:
            # 镜像非对称关节旋转
            mir_body_pose[i*3:(i+1)*3] = torch.tensor(mirror_rot(body_pose[i*3:(i+1)*3]))
    body_pose[:]=mir_body_pose[:]

def add_root(names):
    names = copy.deepcopy(names)
    names.insert(0, 'root')
    return names


def get_global_full_pose(smpl_model, smpl_output, pose2rot=False):
    from smplx import lbs
    from figo_common.math import transform_tensor
    pos = smpl_output.joints[:,:24]
    parents = smpl_model.parents
    full_pose = smpl_output.full_pose
    joints_num = full_pose.shape[1]
    batch_size = full_pose.shape[0]
    if pose2rot:
        rot_mats = lbs.batch_rodrigues(full_pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
    else:
        rot_mats = full_pose.view(batch_size, -1, 3, 3)
    transform_chain = [rot_mats[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_rot = torch.matmul(transform_chain[parents[i]], rot_mats[:,i])
        transform_chain.append(curr_rot)
    rot_mats = torch.stack(transform_chain, dim=1)
    rot_vecs = transform_tensor.rotmat_to_rotvec(rot_mats)
    return pos, rot_vecs