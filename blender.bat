@echo off
set WORKSPACE_HOME=%~dp0
set PYTHONPATH=%PYTHONPATH%;%WORKSPACE_HOME%\packages;%WORKSPACE_HOME%\blender_startup;%WORKSPACE_HOME%\common_packages
set BLENDER_SYSTEM_SCRIPTS=%WORKSPACE_HOME%\blender_startup;%BLENDER_SYSTEM_SCRIPTS%
blender.exe -con --log-level -1 --python %WORKSPACE_HOME%\blender_startup\startup.py