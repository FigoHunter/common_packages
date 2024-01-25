import bpy

class TYPE:
    MESH='MESH'
    CURVE='CURVE'
    SURFACE='SURFACE'
    META='META'
    FONT='FONT'
    ARMATURE='ARMATURE'
    LATTICE='LATTICE'
    EMPTY='EMPTY'
    GPENCIL='GPENCIL'
    CAMERA='CAMERA'
    LIGHT='LIGHT'
    SPEAKER='SPEAKER'
    LIGHT_PROBE='LIGHT_PROBE'

def findObjs(name=None, type=None):
    objs=bpy.data.objects
    ls=[]
    for o in objs:
        match=True
        if name and not o.name == name:
            match=False
        if type and not o.type == type:
            match=False
        if match:
            ls.append(o)
    return ls

def clearObjs(name=None, type=None):
    objs=bpy.data.objects
    ls = findObjs(name, type)
    for o in ls:
        objs.remove(o, do_unlink=True)

def createEmpty(name=None, collection=None):
    bpy.ops.object.empty_add(location=(1,1,1))
    obj=bpy.context.object
    if name:
        obj.name=name
    if collection:
        from .collection import moveCollection
        moveCollection(obj, collection) 
    return obj