import pickle
import uuid as UUID
import os
from . import path

CACHE_DIR = os.path.join(path.WORKSPACE_HOME,'temp','cache')


def cache(data, name=None):
    if name is None:
        name = UUID.uuid4().hex.replace('-','')
    path = os.path.join(CACHE_DIR, name+'.pkl')
    _mkdir(os.path.dirname(path))
    with open(path,'wb') as f:
        pickle.dump(data, f)
    return name

def load_cache(name):
    path = os.path.join(CACHE_DIR, name+'.pkl')
    if not os.path.exists(path):
        raise Exception(f'Cache {name} does not exist')
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def _mkdir(path=None):
    if path is None:
        path = CACHE_DIR
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)