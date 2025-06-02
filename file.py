import os
from typing import Literal
import pymel.core as pm


def save_scene(path):
    import pymel.core as pm
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pm.saveAs(os.path.join(os.path.abspath('.'),path), force=True)


def new_scene(*, force=True, unit:Literal['cm', 'm', 'mm', 'km', 'in', 'ft', 'yd'] = 'm'):
    """
    Create a new scene in Maya with the specified unit.
    """
    if pm.sceneName() and not force:
        return
    pm.newFile(force=force)
    pm.currentUnit(linear=unit)

def import_fbx(path, namespace=None):
    """
    Import an FBX file into the current Maya scene.
    """
    import pymel.core as pm
    if namespace:
        pm.mel.eval(f'FBXImport -f -namespace "{namespace}" -file "{path}"')
    else:
        pm.mel.eval(f'FBXImport -f -file "{path}"')