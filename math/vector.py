import numpy as np

def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector/norm

def clamp_mag(vector, mag=1):
    norm = np.linalg.norm(vector)
    return vector * (mag/norm) if norm > mag else vector

def getAffineVector(vector:np.ndarray):
    return np.insert(vector, len(vector),1)

def projectToPlane(vector, norm):
    norm = normalize(norm)
    vector = vector - np.dot(vector, norm)*norm
    return vector

def getAngle(fromVector, toVector, axis=None):
    if axis is None:
        axis = normalize(np.cross(fromVector, toVector))
    if np.linalg.norm(axis) < 0.0001:
        raise Exception("axis is zero vector")
    fromVector = normalize(projectToPlane(fromVector, axis))
    toVector = normalize(projectToPlane(toVector, axis))
    crossed=np.cross(fromVector, toVector)
    angle = np.arcsin(np.linalg.norm(crossed))
    if np.dot(fromVector, toVector)<0:
        angle = np.pi - angle
    sign = np.sign(np.dot(axis, crossed))
    return sign * angle

def applyMatrix(vector, matrix):
    if np.shape(matrix) == (3,3):
        return matrix @ vector
    if np.shape(matrix) == (4,4):
        return (matrix @ getAffineVector(vector))[0:3]

def xAxis():
    return np.array([1,0,0])

def yAxis():
    return np.array([0,1,0])

def zAxis():
    return np.array([0,0,1])