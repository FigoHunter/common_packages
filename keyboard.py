from enum import Enum
from threading import Lock
from pynput import keyboard
from .action import Action
from threading import Thread, Event
from queue import Queue

class Event(Enum):
    PRESS = 1
    RELEASE = 2

class Keys(Enum):
    KEY_A =                 0x0000000000000000000000000001
    KEY_B =                 0x0000000000000000000000000002
    KEY_C =                 0x0000000000000000000000000004
    KEY_D =                 0x0000000000000000000000000008
    KEY_E =                 0x0000000000000000000000000010
    KEY_F =                 0x0000000000000000000000000020
    KEY_G =                 0x0000000000000000000000000040
    KEY_H =                 0x0000000000000000000000000080
    KEY_I =                 0x0000000000000000000000000100
    KEY_J =                 0x0000000000000000000000000200
    KEY_K =                 0x0000000000000000000000000400
    KEY_L =                 0x0000000000000000000000000800
    KEY_M =                 0x0000000000000000000000001000
    KEY_N =                 0x0000000000000000000000002000
    KEY_O =                 0x0000000000000000000000004000
    KEY_P =                 0x0000000000000000000000008000
    KEY_Q =                 0x0000000000000000000000010000
    KEY_R =                 0x0000000000000000000000020000
    KEY_S =                 0x0000000000000000000000040000
    KEY_T =                 0x0000000000000000000000080000
    KEY_U =                 0x0000000000000000000000100000
    KEY_V =                 0x0000000000000000000000200000
    KEY_W =                 0x0000000000000000000000400000
    KEY_X =                 0x0000000000000000000000800000
    KEY_Y =                 0x0000000000000000000001000000
    KEY_Z =                 0x0000000000000000000002000000

    # 数字键
    KEY_0 =                 0x0000000000000000000004000000
    KEY_1 =                 0x0000000000000000000008000000
    KEY_2 =                 0x0000000000000000000010000000
    KEY_3 =                 0x0000000000000000000020000000
    KEY_4 =                 0x0000000000000000000040000000
    KEY_5 =                 0x0000000000000000000080000000
    KEY_6 =                 0x0000000000000000000100000000
    KEY_7 =                 0x0000000000000000000200000000
    KEY_8 =                 0x0000000000000000000400000000
    KEY_9 =                 0x0000000000000000000800000000

    # 功能键
    F1 =                    0x0000000000000000001000000000
    F2 =                    0x0000000000000000002000000000
    F3 =                    0x0000000000000000004000000000
    F4 =                    0x0000000000000000008000000000
    F5 =                    0x0000000000000000010000000000
    F6 =                    0x0000000000000000020000000000
    F7 =                    0x0000000000000000040000000000
    F8 =                    0x0000000000000000080000000000
    F9 =                    0x0000000000000000100000000000
    F10 =                   0x0000000000000000200000000000
    F11 =                   0x0000000000000000400000000000
    F12 =                   0x0000000000000000800000000000

    # 控制键
    ESC =                   0x0000000000000001000000000000
    TAB =                   0x0000000000000002000000000000
    CAPS_LOCK =             0x0000000000000004000000000000
    SHIFT =                 0x0000000000000008000000000000
    CTRL_L =                0x0000000000000010000000000000
    CTRL_R =                0x0000000000000020000000000000
    ALT_L =                 0x0000000000000040000000000000
    ALT_R =                 0x0000000000000080000000000000
    SPACE =                 0x0000000000000100000000000000
    ENTER =                 0x0000000000000200000000000000
    BACKSPACE =             0x0000000000000400000000000000

    # 导航键
    INSERT =                0x0000000000000800000000000000
    DELETE =                0x0000000000001000000000000000
    HOME =                  0x0000000000002000000000000000
    END =                   0x0000000000004000000000000000
    PAGE_UP =               0x0000000000008000000000000000
    PAGE_DOWN =             0x0000000000010000000000000000

    # 方向键
    LEFT =                  0x0000000000020000000000000000
    RIGHT =                 0x0000000000040000000000000000
    UP =                    0x0000000000080000000000000000
    DOWN =                  0x0000000000100000000000000000

    # 数字键盘
    NUMPAD_DOT =            0x0000000000200000000000000000
    NUMPAD_PLUS =           0x0000000000400000000000000000
    NUMPAD_MINUS =          0x0000000000800000000000000000
    NUMPAD_MULTIPLY =       0x0000000001000000000000000000
    NUMPAD_DIVIDE =         0x0000000002000000000000000000
    SCROLL_LOCK =           0x0000000004000000000000000000
    PAUSE =                 0x0000000008000000000000000000
    PRINT_SCREEN =          0x0000000010000000000000000000
    NUM_LOCK =              0x0000000020000000000000000000

    def __int__(self):
        return self.value

    def __or__(self, value):
        return int(self) | int(value)
    
    def __and__(self, value):
        return int(self) & int(value)

    def __invert__(self):
        return ~int(self)
    
    def __xor__(self, value):
        return int(self) ^ int(value)
    
    def __ror__(self, value):
        return int(value) | int(self)
    
    def __rand__(self, value):
        return int(value) & int(self)
    
    def __rxor__(self, value):
        return int(value) ^ int(self)

    def __repr__(self):
        return f"{self.name}({self.value})"

class ListenerHandler:
    def __init__(self):
        self._on_press_callbacks = {}
        self._on_release_callbacks = {}
        self._listerner = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._current_keys = 0
        self._callback_lock = Lock()
        self._interacting = False
        self._trigger_thread = Thread(target=self._trigger, args=(self._get_current_keys, ), daemon=True)
        self._trigger_job = Queue(1)
        # self._trigger_event = Event()
        # self._trigger_event.set()

    def start(self):
        self.callback_lock.acquire()
        self._listerner.start()
        self._trigger_thread.start()
        self.callback_lock.release()


    def stop(self):
        self.callback_lock.acquire()
        self._listerner.stop()
        self._trigger_job.empty()
        self._trigger_job.put('quit')
        self.callback_lock.release()


    def clear(self):
        self.callback_lock.acquire()
        self._on_press_callbacks.clear()
        self._on_release_callbacks.clear()
        self.callback_lock.release()

    def is_pressed(self, keys):
        self.callback_lock.acquire()
        result = self._current_keys & int(keys)
        self.callback_lock.release()
        return result != 0
        
    def _get_current_keys(self):
        return self._current_keys

    @property
    def on_press_callbacks(self):
        return self._on_press_callbacks
    
    @property
    def on_release_callbacks(self):
        return self._on_release_callbacks
    
    @property
    def current_keys(self):
        return self._current_keys
    
    @property
    def callback_lock(self):
        return self._callback_lock

    def _trigger(self,get_keys):
        while True:
            trigger = self._trigger_job.get()
            self.callback_lock.acquire()
            if trigger == 'quit':
                self.callback_lock.release()
                break
            try:
                trigger(get_keys())
            except Exception as e:
                from traceback import print_exc
                print_exc()
                print(e)
            finally:
                self.callback_lock.release()
                self._trigger_job.task_done()

    def _on_press(self,key):
        code = int(KEYS_MAP.get(key, 0))
        # When a key is pressed, add it to the set we are keeping track of and check if this set is in the dictionary
        self._current_keys = self._current_keys | code
        if self._current_keys in self._on_press_callbacks:
            try:
                if self._trigger_job.unfinished_tasks <= 0 and self._trigger_job.empty():
                    # If the current set of keys are in the mapping, execute the function
                    trigger = self._on_press_callbacks[self._current_keys].trigger
                    self._trigger_job.put(trigger)
            except Exception as e:
                from traceback import print_exc
                print_exc()
                print(e)

    def _on_release(self,key):
        # When a key is released, remove it from the set we are keeping track of and check if this set is in the dictionary
        code = int(KEYS_MAP.get(key, 0))
        if self._current_keys in self._on_release_callbacks:
            try:
                if self._trigger_job.unfinished_tasks <= 0 and self._trigger_job.empty():
                    # If the current set of keys are in the mapping, execute the function
                    trigger = self._on_release_callbacks[self._current_keys].trigger
                    self._trigger_job.put(trigger)
            except Exception as e:
                from traceback import print_exc
                print_exc()
                print(e)

        self._current_keys = self._current_keys & ~code

    def register_callback(self, keys, event:Event, callback, **kwargs):
        keys = int(keys)
        # Register a callback for a specific set of keys
        if event == Event.PRESS:
            if keys not in self._on_press_callbacks:
                self._on_press_callbacks[keys] = Action()
            self._on_press_callbacks[keys].register(callback, **kwargs)
        elif event == Event.RELEASE:
            if keys not in self._on_release_callbacks:
                self._on_release_callbacks[keys] = Action()
            self._on_release_callbacks[keys].register(callback, **kwargs)

    def unregister_callback(self, keys, event:Event, callback):
        keys = int(keys)
        # Unregister a callback for a specific set of keys
        if event == Event.PRESS:
            self._on_press_callbacks[keys] -= callback
        elif event == Event.RELEASE:
            self._on_release_callbacks[keys] -= callback

    def register_op(self, operation, callback, **kwargs):
        self.register_callback(operation.get_key(), operation.get_event(), callback, **kwargs)

    def unregister_op(self, operation, event:Event, callback):
        self.unregister_callback(operation.get_key(),  operation.get_event(), callback)

    def register_key_wrap(self, key, event:Event):
        def wrapper(func):
            self.register_callback(key, event, func)
            return func
        return wrapper
    
    def register_op_wrap(self, operation):
        def wrapper(func):
            self.register_op(operation, func)
            return func
        return wrapper
        

class Event(Enum):
    PRESS = 1
    RELEASE = 2

KEYS_MAP = {
    keyboard.KeyCode.from_char('a'): Keys.KEY_A,
    keyboard.KeyCode.from_char('b'): Keys.KEY_B,
    keyboard.KeyCode.from_char('c'): Keys.KEY_C,
    keyboard.KeyCode.from_char('d'): Keys.KEY_D,
    keyboard.KeyCode.from_char('e'): Keys.KEY_E,
    keyboard.KeyCode.from_char('f'): Keys.KEY_F,
    keyboard.KeyCode.from_char('g'): Keys.KEY_G,
    keyboard.KeyCode.from_char('h'): Keys.KEY_H,
    keyboard.KeyCode.from_char('i'): Keys.KEY_I,
    keyboard.KeyCode.from_char('j'): Keys.KEY_J,
    keyboard.KeyCode.from_char('k'): Keys.KEY_K,
    keyboard.KeyCode.from_char('l'): Keys.KEY_L,
    keyboard.KeyCode.from_char('m'): Keys.KEY_M,
    keyboard.KeyCode.from_char('n'): Keys.KEY_N,
    keyboard.KeyCode.from_char('o'): Keys.KEY_O,
    keyboard.KeyCode.from_char('p'): Keys.KEY_P,
    keyboard.KeyCode.from_char('q'): Keys.KEY_Q,
    keyboard.KeyCode.from_char('r'): Keys.KEY_R,
    keyboard.KeyCode.from_char('s'): Keys.KEY_S,
    keyboard.KeyCode.from_char('t'): Keys.KEY_T,
    keyboard.KeyCode.from_char('u'): Keys.KEY_U,
    keyboard.KeyCode.from_char('v'): Keys.KEY_V,
    keyboard.KeyCode.from_char('w'): Keys.KEY_W,
    keyboard.KeyCode.from_char('x'): Keys.KEY_X,
    keyboard.KeyCode.from_char('y'): Keys.KEY_Y,
    keyboard.KeyCode.from_char('z'): Keys.KEY_Z,

    # 数字键
    keyboard.KeyCode.from_char('0'): Keys.KEY_0,
    keyboard.KeyCode.from_char('1'): Keys.KEY_1,
    keyboard.KeyCode.from_char('2'): Keys.KEY_2,
    keyboard.KeyCode.from_char('3'): Keys.KEY_3,
    keyboard.KeyCode.from_char('4'): Keys.KEY_4,
    keyboard.KeyCode.from_char('5'): Keys.KEY_5,
    keyboard.KeyCode.from_char('6'): Keys.KEY_6,
    keyboard.KeyCode.from_char('7'): Keys.KEY_7,
    keyboard.KeyCode.from_char('8'): Keys.KEY_8,
    keyboard.KeyCode.from_char('9'): Keys.KEY_9,

    # 功能键
    keyboard.Key.f1: Keys.F1,
    keyboard.Key.f2: Keys.F2,
    keyboard.Key.f3: Keys.F3,
    keyboard.Key.f4: Keys.F4,
    keyboard.Key.f5: Keys.F5,
    keyboard.Key.f6: Keys.F6,
    keyboard.Key.f7: Keys.F7,
    keyboard.Key.f8: Keys.F8,
    keyboard.Key.f9: Keys.F9,
    keyboard.Key.f10: Keys.F10,
    keyboard.Key.f11: Keys.F11,
    keyboard.Key.f12: Keys.F12,

    # 控制键
    keyboard.Key.esc: Keys.ESC,
    keyboard.Key.tab: Keys.TAB,
    keyboard.Key.caps_lock: Keys.CAPS_LOCK,
    keyboard.Key.shift: Keys.SHIFT,
    keyboard.Key.ctrl_l: Keys.CTRL_L,
    keyboard.Key.ctrl_r: Keys.CTRL_R,
    keyboard.Key.alt_l: Keys.ALT_L,
    keyboard.Key.alt_r: Keys.ALT_R,
    keyboard.Key.space: Keys.SPACE,
    keyboard.Key.enter: Keys.ENTER,
    keyboard.Key.backspace: Keys.BACKSPACE,

    # 导航键
    keyboard.Key.insert: Keys.INSERT,
    keyboard.Key.delete: Keys.DELETE,
    keyboard.Key.home: Keys.HOME,
    keyboard.Key.end: Keys.END,
    keyboard.Key.page_up: Keys.PAGE_UP,
    keyboard.Key.page_down: Keys.PAGE_DOWN,

    # 方向键
    keyboard.Key.left: Keys.LEFT,
    keyboard.Key.right: Keys.RIGHT,
    keyboard.Key.up: Keys.UP,
    keyboard.Key.down: Keys.DOWN,

    # 数字键盘
    keyboard.KeyCode.from_char('.'): Keys.NUMPAD_DOT,
    keyboard.KeyCode.from_char('+'): Keys.NUMPAD_PLUS,
    keyboard.KeyCode.from_char('-'): Keys.NUMPAD_MINUS,
    keyboard.KeyCode.from_char('*'): Keys.NUMPAD_MULTIPLY,
    keyboard.KeyCode.from_char('/'): Keys.NUMPAD_DIVIDE,

    # 其他常见键
    keyboard.Key.scroll_lock: Keys.SCROLL_LOCK,
    keyboard.Key.pause: Keys.PAUSE,
    keyboard.Key.print_screen: Keys.PRINT_SCREEN,
    keyboard.Key.num_lock: Keys.NUM_LOCK,
}


def get_key_handler():
    return ListenerHandler()

def get_keys(keys):
    list = []
    for key in Keys.__members__.values():
        if keys & key:
            list.append(key)
    return list

class Operation(Enum):
    def get_key(self):
        raise NotImplementedError
    def get_event(self):
        raise NotImplementedError