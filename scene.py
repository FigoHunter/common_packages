import mujoco
import numpy as np

def add_visual_capsule(scene, point1, point2, radius,*, rgba=np.array([1,1,1,1])):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
    mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
    np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
    mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
    point1[0], point1[1], point1[2],
    point2[0], point2[1], point2[2])

def add_axis_aligned_box(scene, minx,miny,minz, maxx, maxy, maxz,*,rgba=np.array([1,1,1,1])):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1

    assert maxx>=minx and maxy>=miny and maxz>=minz

    size = np.array([maxx-minx, maxy-miny, maxz-minz])/2
    pos = np.array([
        np.average([maxx,minx]),
        np.average([maxy,miny]),
        np.average([maxz,minz])])
    print(size)
    print(pos)

    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1], mujoco.mjtGeom.mjGEOM_BOX,size, pos, np.eye(3).reshape(9,1), rgba.astype(np.float32))

def add_visual_box(scene, pos=np.array([0,0,0]), quat=np.array([0,0,0,1]), size=np.array([1,1,1]),rgba=np.array([1,1,1,1])):
    from scipy.spatial.transform import Rotation
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1

    rot_mat = Rotation.from_quat(quat).as_matrix()

    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1], mujoco.mjtGeom.mjGEOM_BOX,size, pos, rot_mat.reshape((9,1)), rgba.astype(np.float32))

