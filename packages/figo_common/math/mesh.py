from . import halfedge
from . import matrix
from . import vector
import numpy as np
from figo_common import math

bbox_points=[0,1,7,2,3,6,4,5]
bbox_face=[[0,2,7,1],[0,3,5,2],[3,0,1,6],[4,5,3,6],[4,6,1,7],[4,7,2,5]]
bbox_halfEdge=halfedge.HalfEdgeObject(bbox_face)

def getMeshByPath(path):
    import open3d as o3d
    return o3d.io.read_triangle_mesh(path)


def getOrientedBboxEdgeVectors(mesh):
    import open3d as o3d
    bbox = mesh.get_minimal_oriented_bounding_box()
    verts = np.asarray(bbox.get_box_points())
    edges = bbox_halfEdge.getEdgesOfVert(0)
    dirs=[verts[e[1]]-verts[e[0]] for e in edges]
    return dirs


def getOrientationOfAlignedY(mesh):
    import open3d as o3d
    from scipy.spatial.transform.rotation import Rotation
    bbox = mesh.get_minimal_oriented_bounding_box()
    verts = np.asarray(bbox.get_box_points())
    edges = bbox_halfEdge.getEdgesOfVert(0)
    yAlignMin=1000
    minDir=None
    for e in edges:
        dir=verts[e[1]]-verts[e[0]]
        align=np.dot(dir,[0,1,0])
        if abs(align) < abs(yAlignMin):
            yAlignMin = align
            minDir = vector.normalize(dir)
    minDir[1]=0
    minDir = vector.normalize(minDir)
    print(f'mindir:{minDir}')
    crossed=np.cross(minDir,np.array([1,0,0]))
    print(f'crossed:{crossed}')
    angle =  np.arcsin(np.linalg.norm(crossed))
    print(f'angle:{angle}')
    rot = Rotation.from_euler('y',angle,degrees=False)
    return rot.as_matrix()

def getAxisAlignedBbox(mesh, matrix = np.eye(4)):
    import open3d as o3d
    import copy
    matrix=math.matrix.getAffineMat(matrix)
    mesh = copy.deepcopy(mesh)
    mesh.transform(matrix)
    bbox = mesh.get_axis_aligned_bounding_box()
    return bbox.get_min_bound(), bbox.get_max_bound()


def getMinOrientedBbox(mesh, matrix = np.eye(4)):
    import open3d as o3d
    import copy
    matrix=math.matrix.getAffineMat(matrix)
    mesh = copy.deepcopy(mesh)
    mesh.transform(matrix)
    bbox = mesh.get_minimal_oriented_bounding_box()
    verts = np.asarray(bbox.get_box_points())
    edges = bbox_halfEdge.getEdgesOfVert(0)
    dirs=(verts[e[1]]-verts[e[0]] for e in edges)
    return verts[0], dirs