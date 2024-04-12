import bpy

def getBones(object):
    if object.type == 'ARMATURE':
        return object.data.bones
    return None

def getBone(object, name):
    return getBones(object).get(name)

def getPoseBones(object):
    if object.type == 'ARMATURE':
        return object.pose.bones
    return None

def getPoseBone(object, name):
    return getPoseBones(object).get(name)


        