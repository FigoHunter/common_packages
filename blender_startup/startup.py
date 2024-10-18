import importlib
import os
import pkgutil
from figo_common import path as fpath
from figo_common.module import runtime_import

dir = fpath.STARTUP_PATH

for f in os.listdir(dir):
    path = os.path.join(dir, f)
    try:
        if os.path.isdir(path):
            runtime_import(path)
            print(f'Imported {path}')
    except Exception as e:
        print(f'Skipped importing {path}: {e}')
        continue
