import importlib

def module_exists(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False