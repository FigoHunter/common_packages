#!/bin/bash
export WORKSPACE_HOME=$(dirname $(readlink -f "$0"))
export PYTHONPATH=${PYTHONPATH}:${WORKSPACE_HOME}/packages:${WORKSPACE_HOME}/blender_startup
export BLENDER_SYSTEM_SCRIPTS=${WORKSPACE_HOME}/blender_startup:${BLENDER_SYSTEM_SCRIPTS}
echo ${PYTHONPATH}
echo ${BLENDER_SYSTEM_SCRIPTS}
blender -con --log-level -1 --python ${WORKSPACE_HOME}/blender_startup/startup.py

