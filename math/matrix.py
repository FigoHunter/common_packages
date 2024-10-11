import numpy as np
from scipy.spatial.transform import Rotation

def getAffineMat(matrix:np.ndarray) -> np.ndarray:
    if np.shape(matrix) == (4,4):
        return matrix
    assert np.shape(matrix) == (3,3)
    affine_mat = matrix.copy()
    affine_mat = np.insert(affine_mat,3,values=[0,0,0],axis=1)
    affine_mat = np.insert(affine_mat,3,values=[0,0,0,1],axis=0)
    return affine_mat

def getAffineMatFromTransl(x,y,z) -> np.ndarray:
    return np.array([[1,0,0,x],
                     [0,1,0,y],
                     [0,0,1,z],
                     [0,0,0,1]])

def getAffineMatFromTranslArray(v:np.ndarray) -> np.ndarray:
    return getAffineMatFromTransl(v[0],v[1],v[2])

def decomposeTrs(mat):
    import transformations
    scale, shear, angles, transl, persp = transformations.decompose_matrix(mat)
    quat = transformations.quaternion_from_euler(*angles)
    quat = np.array([quat[1],quat[2],quat[3],quat[0]])
    return transl, quat, scale

def getAffineMatFromQuat(quat):
    return getAffineMat(Rotation.from_quat(quat).as_matrix())

def getAffineMatFromEuler(euler, order='xyz', degree=False):
    return getAffineMat(Rotation.from_euler(order,euler,degree).as_matrix())

def getQuaternionFromMatrix(mat):
    import transformations
    quat= transformations.quaternion_from_matrix(mat,True)
    return np.array([quat[1],quat[2],quat[3],quat[0]])

def compute_swing_rotation(v1, v2):
    """
    计算将向量 v1 旋转到向量 v2 的 Swing 旋转矩阵。

    参数:
    - v1: 初始方向向量，形状为 (3,)
    - v2: 目标方向向量，形状为 (3,)

    返回:
    - swing_matrix: Swing 旋转矩阵，形状为 (3, 3)
    """

    # 规范化输入向量
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # 计算旋转轴（垂直于 v1 和 v2 的向量）
    rot_axis = np.cross(v1_norm, v2_norm)
    axis_norm = np.linalg.norm(rot_axis)

    if axis_norm < 1e-8:
        # 向量平行或反向，无需旋转或旋转180度
        if np.dot(v1_norm, v2_norm) > 0:
            # 同方向，返回单位矩阵
            swing_matrix = np.eye(3)
        else:
            # 反方向，绕任意垂直于 v1 的轴旋转180度
            # 选择一个垂直于 v1 的向量作为旋转轴
            if abs(v1_norm[0]) < abs(v1_norm[1]):
                temp_axis = np.array([1, 0, 0])
            else:
                temp_axis = np.array([0, 1, 0])
            rot_axis = np.cross(v1_norm, temp_axis)
            rot_axis /= np.linalg.norm(rot_axis)
            swing_matrix = rotation_matrix_axis_angle(rot_axis, np.pi)
    else:
        # 规范化旋转轴
        rot_axis /= axis_norm

        # 计算旋转角度
        cos_theta = np.dot(v1_norm, v2_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # 构建旋转矩阵
        swing_matrix = rotation_matrix_axis_angle(rot_axis, theta)

    return swing_matrix

def rotation_matrix_axis_angle(axis, angle):
    """
    根据旋转轴和角度构建旋转矩阵。

    参数:
    - axis: 旋转轴，形状为 (3,)
    - angle: 旋转角度，标量

    返回:
    - rotation_matrix: 旋转矩阵，形状为 (3, 3)
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    rotation_matrix = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ])

    return rotation_matrix