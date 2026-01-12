import pygame
from .fsm import *
import numpy as np

class AxisMapping:
    LEFT_STICK_X = 0
    LEFT_STICK_Y = 1
    LEFT_TRIGGER = 2
    RIGHT_STICK_X = 3
    RIGHT_STICK_Y = 4
    RIGHT_TRIGGER = 5

class ButtonMapping:
    XBOX_A = 0
    XBOX_B = 1
    XBOX_Y = 2
    XBOX_X = 3
    PS_CROSS = 0
    PS_CIRCLE = 1
    PS_TRIANGLE = 2
    PS_SQUARE = 3
    LEFT_BUMPER = 4
    RIGHT_BUMPER = 5
    SELECT = 8
    START = 9
    LOGO = 10
    LEFT_STICK_PRESS = 11
    RIGHT_STICK_PRESS = 12

class HatMapping:
    LEFT = np.asarray((-1, 0))
    RIGHT = np.asarray((1, 0))
    UP = np.asarray((0, 1))
    DOWN = np.asarray((0, -1))

def init_joystick():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick


_button_pressed = False
def get_button_single_tap(joystick, button):
    global _button_pressed
    value = joystick.get_button(button) > 0.5
    if value:
        if _button_pressed:
            out = False
        else:
            out = True
    else:
        out = False
    _button_pressed = value    
    return out

def get_axis(joystick, axis, deadzone=0.1):
    value = joystick.get_axis(axis)
    if abs(value) < deadzone:
        return 0.0
    return value

def get_axis_command(joystick, deadzone = 0.1):
    return -get_axis(joystick, AxisMapping.LEFT_STICK_Y, deadzone), -get_axis(joystick, AxisMapping.LEFT_STICK_X, deadzone), -get_axis(joystick, AxisMapping.RIGHT_STICK_X, deadzone)

class JoystickStateMachine(FiniteStateMachine):
    def __init__(self):
        super().__init__()
        self.joystick = init_joystick()
        self.early_update = Action(self.joysticks_update).action

    def joysticks_update(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.dispatch_event(Event('axis', event))
            if event.type == pygame.JOYBUTTONDOWN:
                self.dispatch_event(Event('button', event))
            if event.type == pygame.JOYHATMOTION:
                self.dispatch_event(Event('hat', event))

    def get_button_single_tap(self, button):
        return get_button_single_tap(self.joystick, button)
    
    def get_axis(self, axis, deadzone=0.1):
        return get_axis(self.joystick, axis, deadzone)
    
    def get_axis_command(self, deadzone=0.1):
        return get_axis_command(self.joystick, deadzone)
    
    def get_hat(self, hat:HatMapping):
        value = np.asarray(self.joystick.get_hat(0))
        flag = value * hat
        return not np.all(flag < 1)
