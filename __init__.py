from . import environ
from . import file
from . import module
from . import path
from . import cache

def get_uuid():
    import uuid
    return uuid.uuid4().hex.replace('-','')

def load_json(path):
    import json
    with open(path,'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    import json
    with open(path,'w') as f:
        json.dump(data, f)

def load_np(path):
    import numpy as np
    return np.load(path)

def save_np(path, data):
    import numpy as np
    np.save(path, data)

def load_pkl(path):
    import pickle
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(path, data):
    import pickle
    with open(path,'wb') as f:
        pickle.dump(data, f)