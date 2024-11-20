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

def findObjs(name=None, type=None, collection = None):
    if collection is None:
        objs=bpy.data.objects
    else:
        if isinstance(collection, str):
            collection = bpy.data.collections.get(collection, None)
        if collection is None:
            return []
        objs=collection.objects
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

def create_sphere(name=None, radius=1, location=(0,0,0), collection=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj=bpy.context.object
    if name:
        obj.name=name
    if collection:
        from .collection import moveCollection
        moveCollection(obj, collection) 
    return obj

def get_or_create_axis(name, collection=None, display_type='SPHERE', size=0.1):
    obj = findObjs(name=name, collection=collection)
    if obj:
        return obj[0]
        # Create a new empty object
    obj = bpy.data.objects.new(name=name, object_data=None)
    obj.empty_display_type = display_type
    obj.location = (0, 0, 0)
    obj.empty_display_size = size

    if collection:
        from .collection import moveCollection
        moveCollection(obj, collection) 
    return obj