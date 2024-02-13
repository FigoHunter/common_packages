from ._base import load_json
import sys
import importlib
import os

def import_extern(name):
    packages = load_json('extern.json', {})
    if name in packages:
        pkgs = packages[name]['packages']
        for pkg in pkgs:
            path = os.path.dirname(pkg)
            pkg_name = os.path.basename(pkg)
            sys.path.append(path)
            imported = importlib.import_module(pkg_name)
            print(f'{name}.{pkg_name}')
            globals()[f'{name}_{pkg_name}'] = imported