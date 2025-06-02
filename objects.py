import pymel.core as pm

def create_locator(*, name='something', pos=[0, 0, 0], orient=[0, 0, 0], parent=None):
    """
    Create a locator in the scene with the given name, position, and orientation.
    Optionally, you can specify a parent for the locator.
    """
    loc = pm.spaceLocator(name=name)
    loc.setTranslation(pos)
    loc.setRotation(orient)
    if parent:
        pm.parent(loc, parent)
    return loc
