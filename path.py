import os
from . import environ

WORKSPACE_HOME=environ.getEnvVar("WORKSPACE_HOME")
if not WORKSPACE_HOME:
    WORKSPACE_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
DATA_PATH = os.path.join(WORKSPACE_HOME, 'data')
ASSET_PATH = os.path.join(WORKSPACE_HOME, 'assets')
STARTUP_PATH=os.path.join(WORKSPACE_HOME,"startup")


def setWorkspaceHome(path):
    global WORKSPACE_HOME
    WORKSPACE_HOME=path