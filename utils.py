

def mel_cmds(cmds):
    import pymel.core as pm
    import maya.mel as mel
    values = []
    if isinstance(cmds, str):
        cmds = [cmds]
    for cmd in cmds:
        values.append(mel.eval(cmd))
    return values[0] if len(values) == 1 else values

def _is_under(child, root) -> bool:
    node = child
    while node:
        if node == root:
            return True
        node = node.getParent()
    return False


def find_joint(name: str, root: str = None):
    import pymel.core as pm
    try:
        joint = pm.PyNode(name)
    except pm.MayaNodeError:
        raise ValueError(f"Joint '{name}' not found")
    if joint.type() != 'joint':
        raise ValueError(f"Node '{name}' is not a joint")
    if root is None:
        return joint
    try:
        root_joint = pm.PyNode(root)
    except pm.MayaNodeError:
        raise ValueError(f"Root '{root}' not found")
    if root_joint.type() != 'joint':
        raise ValueError(f"Root '{root}' is not a joint")

    if _is_under(joint, root_joint):
        return joint
    raise ValueError(f"Joint '{name}' is not a child of root '{root}'")


def rebake(*,root_name=None, start=None, end=None):
    import pymel.core as pm
    from .playback import start_frame, end_frame
    if start is None:
        start = start_frame()
    if end is None:
        end = end_frame()
    if root_name is None:
        joints = pm.ls(type='joint')
    else:
        root = pm.ls(root_name, type='joint')[0]
        joints = [root] + pm.listRelatives(root, ad=True, type='joint')
    pm.bakeResults(
        joints,
        simulation=True,
        t=(start, end),
        sampleBy=1,
        disableImplicitControl=True,
        preserveOutsideKeys=False,
)