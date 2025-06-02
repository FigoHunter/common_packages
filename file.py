import numpy as np
import torch
import pickle
import os

def save(params, path):
    for key, value in params.items():
        assert isinstance(value, torch.Tensor), f"Value for {key} is not a tensor"
        params[key] = value.detach().cpu().numpy()
    with open(path, 'wb') as f:
        pickle.dump(params, f)

def load(path, *, params=None, dtype=torch.float32, device='cpu'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if params is None:
        params = {}
    for key, value in data.items():
        set_in_place = False
        if key in params and params[key] is not None:
            assert isinstance(params[key], torch.Tensor), f"Value for {key} is not a tensor"
            assert isinstance(value, np.ndarray), f"Value for {key} is not a numpy array"
            if value.shape == tuple(params[key].shape):
                params[key].copy_(torch.from_numpy(value))
                set_in_place = True
        if not set_in_place:
            params[key] = torch.tensor(value, device=device, dtype=dtype)
            
    return params