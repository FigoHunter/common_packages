import pymel.core as pm
from typing import Literal

def start_frame(frame = None):
    # get animaton real start
    if frame is None:
        return pm.playbackOptions(q=True, ast=True)
    else:
        pm.playbackOptions(ast=frame)
        return frame
    
def end_frame(frame = None):
    # get animation real end
    if frame is None:
        return pm.playbackOptions(q=True, aet=True)
    else:
        pm.playbackOptions(aet=frame)
        return frame

def current_frame(frame = None):
    # get current frame
    if frame is None:
        return pm.currentTime(q=True)
    else:
        pm.currentTime(frame)
        return frame

_FPS_TO_UNIT = {
    24: "film", 25: "pal", 30: "ntsc", 50: "50fps",
    60: "60fps", 100: "100fps", 120: "120fps",
}

def set_fps(fps: Literal[24,25,30,50,60,100,120]) -> None:
    unit = _FPS_TO_UNIT[fps]
    if pm.currentUnit(q=True, time=True) != unit:
        pm.currentUnit(time=unit)
    end = end_frame()
    end_frame(int(end))