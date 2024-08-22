import importlib
import importlib.util
import sys

def module_exists(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    
def runtime_import(path):
    import os
    module_name = os.path.basename(path)
    init_py = os.path.join(path, '__init__.py')
    spec = importlib.util.spec_from_file_location(module_name, init_py)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo