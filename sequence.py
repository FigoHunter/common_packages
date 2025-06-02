import smplx
import torch
from smplx_figo import file as sfile


def frames_count_param(params, param_name):
    if param_name in params:
        return params[param_name].shape[0]
    return 0

def frames_count(params):
    frames = 0
    frames = max(frames, frames_count_param(params, 'transl'))
    frames = max(frames, frames_count_param(params, 'body_pose'))
    frames = max(frames, frames_count_param(params, 'global_orient'))
    frames = max(frames, frames_count_param(params, 'expression'))
    frames = max(frames, frames_count_param(params, 'left_hand_pose'))
    frames = max(frames, frames_count_param(params, 'right_hand_pose'))
    frames = max(frames, frames_count_param(params, 'jaw_pose'))
    frames = max(frames, frames_count_param(params, 'leye_pose'))
    frames = max(frames, frames_count_param(params, 'reye_pose'))
    return frames

def _insert_param(params, param_name, frames, assert_frames):
    frame_count = frames_count_param(params, param_name)
    if frame_count != assert_frames:
        return params
    param = params[param_name]
    inserted = torch.zeros((frames,) + param.shape[1:], device=param.device, dtype=param.dtype)
    for i in range(frames):
        inserted[i] = inserted[i] + (param[0] - inserted[i]) * (i / frames)
    param = torch.concat([inserted, param], dim=0)
    params[param_name] = param
    return params

def _abort_param_frames(params, param_name, frames, assert_frames):
    frame_count = frames_count_param(params, param_name)
    if frame_count != assert_frames:
        return params
    if frame_count < len(frames):
        raise ValueError(f"Frame count {frame_count} is less than {frames}")
    param = params[param_name]
    frames = [*set(frames)]
    frames.sort()
    shape = param.shape
    shape = shape[0] - len(frames), *shape[1:]
    new_param = torch.zeros(shape, device=param.device, dtype=param.dtype)
    pre_f = 0
    count = 0
    for f in frames:
        if pre_f-1 == f:
            raise ValueError(f"Frame {f} appears multiple times")
        new_param[count:count+f-pre_f] = param[pre_f:f]
        count += f - pre_f
        pre_f = f + 1
    new_param[count:] = param[pre_f:]

    params[param_name] = new_param
    return params

def _insert_same_param(params, param_name, frames, assert_frames):
    frame_count = frames_count_param(params, param_name)
    if frame_count != assert_frames:
        return params
    param = params[param_name]
    inserted = torch.zeros((frames,) + param.shape[1:], device=param.device, dtype=param.dtype)
    for i in range(frames):
        inserted[i] = param[0]
    param = torch.concat([inserted, param], dim=0)
    params[param_name] = param
    return params

@torch.no_grad()
def insert_smpl_tpose(params, frames):
    frame_count = frames_count(params)
    if frame_count == 0:
        return params
    params = _insert_param(params, 'transl', frames, frame_count)
    params = _insert_param(params, 'body_pose', frames, frame_count)
    params = _insert_param(params, 'global_orient', frames, frame_count)
    params = _insert_param(params, 'expression', frames, frame_count)
    params = _insert_param(params, 'left_hand_pose', frames, frame_count)
    params = _insert_param(params, 'right_hand_pose', frames, frame_count)
    params = _insert_param(params, 'jaw_pose', frames, frame_count)
    params = _insert_param(params, 'leye_pose', frames, frame_count)
    params = _insert_param(params, 'reye_pose', frames, frame_count)
    params = _insert_same_param(params, 'betas', frames, frame_count)
    return params

@torch.no_grad()
def remove_frames(params, frames=[]):
    frame_count = frames_count(params)
    if frame_count == 0:
        return params
    params = _abort_param_frames(params, 'transl', frames, frame_count)
    params = _abort_param_frames(params, 'body_pose', frames, frame_count)
    params = _abort_param_frames(params, 'global_orient', frames, frame_count)
    params = _abort_param_frames(params, 'expression', frames, frame_count)
    params = _abort_param_frames(params, 'left_hand_pose', frames, frame_count)
    params = _abort_param_frames(params, 'right_hand_pose', frames, frame_count)
    params = _abort_param_frames(params, 'jaw_pose', frames, frame_count)
    params = _abort_param_frames(params, 'leye_pose', frames, frame_count)
    params = _abort_param_frames(params, 'reye_pose', frames, frame_count)
    params = _abort_param_frames(params, 'betas', frames, frame_count)
    return params

# if __name__ == '__main__':
#     test_path = './pipeline/1.process_mocap/data/4/upbody_1363-2552_smpl.pkl'
#     data = sfile.load(test_path, dtype=torch.float32, device='cpu')
#     data = insert_smpl_tpose(data, 100)
#     sfile.save(data, './test.pkl')
    

    
    
