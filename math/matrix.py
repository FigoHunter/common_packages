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