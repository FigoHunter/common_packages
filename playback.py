import pymel.core as pm
from typing import Literal

def start_frame(frame = None):
    # get animaton real start
    if frame is None:
        return pm.playbackOptions(q=True, ast=True)
    else:
        pm.playbackOptions(ast=frame, min=frame)  # 同时设置 animation start 和 playback min
        return frame
    
def end_frame(frame = None):
    # get animation real end
    if frame is None:
        return pm.playbackOptions(q=True, aet=True)
    else:
        pm.playbackOptions(aet=frame, max=frame)  # 同时设置 animation end 和 playback max
        return frame

def current_frame(frame = None):
    # get current frame
    if frame is None:
        return pm.currentTime(q=True)
    else:
        pm.currentTime(frame)
        return frame

# Maya 支持的帧率映射
# 参考: https://help.autodesk.com/cloudhelp/2024/CHS/Maya-Tech-Docs/CommandsPython/currentUnit.html
_FPS_TO_UNIT = {
    2: "2fps",
    3: "3fps",
    4: "4fps",
    5: "5fps",
    6: "6fps",
    8: "8fps",
    10: "10fps",
    12: "12fps",
    15: "game",       # 15 fps
    16: "16fps",
    20: "20fps",
    23.976: "23.976fps",
    24: "film",       # 24 fps
    25: "pal",        # 25 fps (PAL/SECAM)
    29.97: "29.97fps",
    30: "ntsc",       # 30 fps (NTSC)
    40: "40fps",
    47.952: "47.952fps",
    48: "48fps",
    50: "palf",       # 50 fps (PAL Field)
    59.94: "59.94fps",
    60: "ntscf",      # 60 fps (NTSC Field)
    75: "75fps",
    80: "80fps",
    90: "90fps",
    100: "100fps",
    120: "120fps",
    125: "125fps",
    150: "150fps",
    200: "200fps",
    240: "240fps",
    250: "250fps",
    300: "300fps",
    400: "400fps",
    500: "500fps",
    600: "600fps",
    750: "750fps",
    1200: "1200fps",
    1500: "1500fps",
    2000: "2000fps",
    3000: "3000fps",
    6000: "6000fps",
}

# 常用帧率类型提示
FpsLiteral = Literal[
    2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20,
    23.976, 24, 25, 29.97, 30, 40, 47.952, 48,
    50, 59.94, 60, 75, 80, 90, 100, 120,
    125, 150, 200, 240, 250, 300, 400, 500, 600, 750,
    1200, 1500, 2000, 3000, 6000
]

def set_fps(fps: FpsLiteral) -> None:
    """
    设置 Maya 场景帧率。
    
    Args:
        fps: 帧率值，必须是 Maya 支持的帧率之一。
             常用值: 24 (film), 25 (pal), 30 (ntsc), 50, 60, 120
    
    Raises:
        ValueError: 如果帧率不被 Maya 支持
    """
    if fps not in _FPS_TO_UNIT:
        supported = sorted(_FPS_TO_UNIT.keys())
        raise ValueError(f"Maya 不支持帧率 {fps}。支持的帧率: {supported}")
    
    unit = _FPS_TO_UNIT[fps]
    if pm.currentUnit(q=True, time=True) != unit:
        pm.currentUnit(time=unit)
    end = end_frame()
    end_frame(int(end))