
def create_joint_world(name, pos, parent=None):
    import pymel.core as pm
    """
    Create a joint in Maya with the specified name, position, rotation matrix, and parent.
    """
    # Create the joint
    joint = pm.joint(name=name, p=pos.tolist(), radius=0.1, a=True)
    # pm.setAttr(f'{joint}.jointOrient', *R.from_matrix(rotm).as_euler('xyz', degrees=True))
    pm.parent(joint, parent)
    return joint

