import os
from . import environ

WORKSPACE_HOME=environ.getEnvVar("WORKSPACE_HOME")
if not WORKSPACE_HOME:
    WORKSPACE_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
DATA_PATH = os.path.join(WORKSPACE_HOME, 'data')
ASSET_PATH = os.path.join(WORKSPACE_HOME, 'assets')
STARTUP_PATH=os.path.join(WORKSPACE_HOME,"startup")


def set_workspace_home(path):
    global WORKSPACE_HOME
    WORKSPACE_HOME=path

def enumerate_path(path, full_path=False, file=True, dir=True):
    path = os.path.abspath(path)
    for subpath in os.listdir(path):
        abspath = os.path.join(path, subpath)
        if dir and os.path.isdir(abspath):
            if full_path:
                yield abspath
            else:
                yield subpath
        elif file and os.path.isfile(abspath):
            if full_path:
                yield abspath
            else:
                yield subpath
