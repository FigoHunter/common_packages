import os
import platform
from typing import List

def getEnvVar(key:str, default=''):
    return os.environ.get(key,default)

def getEnvVarAsList(key:str):
    value = getEnvVar(key)
    if not value:
        return []
    if platform.system().lower() == 'windows':
        return value.split(";")
    elif platform.system().lower() == 'linux':
        return value.split(":")
    else:
        raise Exception(f"系统未支持：{platform.system().lower()}")

def setEnvVar(key:str, value:str):
    os.environ[key] = value

def setEnvVarList(key:str, values:List[str]):
    if platform.system().lower() == 'windows':
        value = ";".join(values)
    elif platform.system().lower() == 'linux':
        value = ":".join(values)
    else:
        raise Exception(f"系统未支持：{platform.system().lower()}")
    setEnvVar(key, value)