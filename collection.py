import bpy

def getOrNewCollection(name):
    if name not in bpy.data.collections:
        coll = bpy.data.collections.new(name)
        sceneCollection().children.link(coll)
        return coll
    return bpy.data.collections[name]

def removeCollection(name, withobject=False):
    if name not in bpy.data.collections:
        return
    collection = bpy.data.collections.get(name)
    if withobject:
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)

def moveCollection(obj, collection):
    if isinstance(collection,str):
        collection=getOrNewCollection(collection)
    flag = True
    for old in obj.users_collection:
        if old != collection:
            old.objects.unlink(obj)
        else:
            flag=False
    if flag:
        collection.objects.link(obj)

def sceneCollection():
    return bpy.context.scene.collection