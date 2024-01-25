import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import threading

from glfw import KEY_UNKNOWN
from glfw import KEY_SPACE
from glfw import KEY_APOSTROPHE
from glfw import KEY_COMMA
from glfw import KEY_MINUS
from glfw import KEY_PERIOD
from glfw import KEY_SLASH
from glfw import KEY_0
from glfw import KEY_1
from glfw import KEY_2
from glfw import KEY_3
from glfw import KEY_4
from glfw import KEY_5
from glfw import KEY_6
from glfw import KEY_7
from glfw import KEY_8
from glfw import KEY_9
from glfw import KEY_SEMICOLON
from glfw import KEY_EQUAL
from glfw import KEY_A
from glfw import KEY_B
from glfw import KEY_C
from glfw import KEY_D
from glfw import KEY_E
from glfw import KEY_F
from glfw import KEY_G
from glfw import KEY_H
from glfw import KEY_I
from glfw import KEY_J
from glfw import KEY_K
from glfw import KEY_L
from glfw import KEY_M
from glfw import KEY_N
from glfw import KEY_O
from glfw import KEY_P
from glfw import KEY_Q
from glfw import KEY_R
from glfw import KEY_S
from glfw import KEY_T
from glfw import KEY_U
from glfw import KEY_V
from glfw import KEY_W
from glfw import KEY_X
from glfw import KEY_Y
from glfw import KEY_Z
from glfw import KEY_LEFT_BRACKET
from glfw import KEY_BACKSLASH
from glfw import KEY_RIGHT_BRACKET
from glfw import KEY_GRAVE_ACCENT
from glfw import KEY_WORLD_1
from glfw import KEY_WORLD_2
from glfw import KEY_ESCAPE
from glfw import KEY_ENTER
from glfw import KEY_TAB
from glfw import KEY_BACKSPACE
from glfw import KEY_INSERT
from glfw import KEY_DELETE
from glfw import KEY_RIGHT
from glfw import KEY_LEFT
from glfw import KEY_DOWN
from glfw import KEY_UP
from glfw import KEY_PAGE_UP
from glfw import KEY_PAGE_DOWN
from glfw import KEY_HOME
from glfw import KEY_END
from glfw import KEY_CAPS_LOCK
from glfw import KEY_SCROLL_LOCK
from glfw import KEY_NUM_LOCK
from glfw import KEY_PRINT_SCREEN
from glfw import KEY_PAUSE
from glfw import KEY_F1
from glfw import KEY_F2
from glfw import KEY_F3
from glfw import KEY_F4
from glfw import KEY_F5
from glfw import KEY_F6
from glfw import KEY_F7
from glfw import KEY_F8
from glfw import KEY_F9
from glfw import KEY_F10
from glfw import KEY_F11
from glfw import KEY_F12
from glfw import KEY_F13
from glfw import KEY_F14
from glfw import KEY_F15
from glfw import KEY_F16
from glfw import KEY_F17
from glfw import KEY_F18
from glfw import KEY_F19
from glfw import KEY_F20
from glfw import KEY_F21
from glfw import KEY_F22
from glfw import KEY_F23
from glfw import KEY_F24
from glfw import KEY_F25
from glfw import KEY_KP_0
from glfw import KEY_KP_1
from glfw import KEY_KP_2
from glfw import KEY_KP_3
from glfw import KEY_KP_4
from glfw import KEY_KP_5
from glfw import KEY_KP_6
from glfw import KEY_KP_7
from glfw import KEY_KP_8
from glfw import KEY_KP_9
from glfw import KEY_KP_DECIMAL
from glfw import KEY_KP_DIVIDE
from glfw import KEY_KP_MULTIPLY
from glfw import KEY_KP_SUBTRACT
from glfw import KEY_KP_ADD
from glfw import KEY_KP_ENTER
from glfw import KEY_KP_EQUAL
from glfw import KEY_LEFT_SHIFT
from glfw import KEY_LEFT_CONTROL
from glfw import KEY_LEFT_ALT
from glfw import KEY_LEFT_SUPER
from glfw import KEY_RIGHT_SHIFT
from glfw import KEY_RIGHT_CONTROL
from glfw import KEY_RIGHT_ALT
from glfw import KEY_RIGHT_SUPER
from glfw import KEY_MENU
from glfw import KEY_LAST

__viewer_callback_manager={}

class KeyCallbackManager:
    def __init__(self):
        self.viewer=None
        self.__key_pressed=set()
        self.__key_pressed_processing_pool=set()
        self.__lock = threading.Lock()

    def callback(self, keycode:int):
        self.__lock.acquire()
        self.__key_pressed.add(keycode)
        self.__lock.release()

    def flush(self):
        self.__lock.acquire()
        self.__key_pressed_processing_pool.clear()
        self.__lock.release()

    def get_key(self, keycode):
        self.__lock.acquire()
        if keycode in self.__key_pressed_processing_pool:
            value = True
        else:
            value = False
        self.__lock.release()
        return value
    
    def push_processing(self):
        self.__lock.acquire()
        self.__key_pressed_processing_pool.update(self.__key_pressed)
        self.__key_pressed.clear()
        self.__lock.release()



def get_key_pressed(viewer:mujoco.viewer.Handle, keycode):
    cm = __viewer_callback_manager.get(viewer, None)
    if cm is None:
        raise Exception("viewer not launched with mujoco_figo.launch_passive_with_callback")
    pressed=True
    return cm.get_key(keycode)
    
def sync(viewer:mujoco.viewer.Handle):
    cm = __viewer_callback_manager.get(viewer, None)
    if cm is not None:
        cm.flush()
    viewer.sync()
    if cm is not None:
        cm.push_processing()
    


def launch_passive_with_callback(model:mujoco.MjModel, data:mujoco.MjData):
    cm = KeyCallbackManager()
    viewer= mujoco.viewer.launch_passive(model, data, key_callback=cm.callback)
    cm.viewer = viewer
    __viewer_callback_manager[viewer] = cm
    return viewer

def get_body_selected(viewer:mujoco.viewer.Handle, model:mujoco.MjModel):
    id=viewer.perturb.select
    if id < 0:
        return None
    return model.body(id)

    