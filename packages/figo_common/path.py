import os
from . import environ

WORKSPACE_HOME=environ.getEnvVar("WORKSPACE_HOME")
DATA_PATH = os.path.join(WORKSPACE_HOME, 'data')
ASSET_PATH = os.path.join(WORKSPACE_HOME, 'assets')
STARTUP_PATH=os.path.join(WORKSPACE_HOME,"startup")


def setWorkspaceHome(path):
    global WORKSPACE_HOME
    WORKSPACE_HOME=path