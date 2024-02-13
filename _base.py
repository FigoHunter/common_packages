def get_uuid():
    import uuid
    return uuid.uuid4().hex.replace('-','')

def load_json(path, default = None):
    import json
    import os
    if not os.path.exists(path):
        return default
    with open(path,'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    import json
    import os
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'w') as f:
        json.dump(data, f, indent=4)

def load_np(path):
    import numpy as np
    return np.load(path)

def save_np(path, data):
    import numpy as np
    import os
    os.makedirs(os.path.dirname(path),exist_ok=True)
    np.save(path, data)

def load_pkl(path, default = None):
    import pickle
    import os
    if not os.path.exists(path):
        return default
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(path, data):
    import pickle
    import os
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'wb') as f:
        pickle.dump(data, f)