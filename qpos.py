import numpy as np

def get_qpos_pos_rot(qpos):
    assert len(qpos)==4 or len(qpos)==7
    if len(qpos) == 4:
        pos = [0,0,0]
        quat = [qpos[1],qpos[2],qpos[3],qpos[0]]
    elif len(qpos) == 7:
        pos = [qpos[0],qpos[1],qpos[2]]
        quat = [qpos[4],qpos[5],qpos[6],qpos[3]]
    return np.array(pos), np.array(quat)

def get_qpos_matrix(qpos):
    import transformations
    from scipy.spatial.transform import Rotation
    pos,quat = get_qpos_pos_rot(qpos)
    euler = Rotation.from_quat(quat).as_euler('xyz',degrees=False)
    mat = transformations.compose_matrix(translate=pos,angles=euler)
    return mat

def identity():
    return [0,0,0,1,0,0,0]