from abc import ABC, abstractmethod
import time

class Action:
    def __init__(self, action, *args, **kwargs):
        self.method = action
        self.args = args
        self.kwargs = kwargs

    def action(self):
        self.method(*self.args, **self.kwargs)

class Event:
    def __init__(self, type, data = None):
        self.type = type
        self.data = data

class State(ABC):
    def __init__(self):
        self._fsm:FiniteStateMachine = None

    @property
    def fsm(self):
        return self._fsm

    @property
    def fsm(self):
        return self._fsm

    @abstractmethod
    def on_enter(self):
        pass

    @abstractmethod
    def on_exit(self):
        pass

    @abstractmethod
    def on_update(self):
        pass

    def on_pre_update(self):
        pass

    def on_post_update(self):
        pass

class EmptyState(State):
    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def on_update(self):
        pass


class FiniteStateMachine:
    def __init__(self):
        self._events_suscribers = {}
        self._state = EmptyState()
        self._events_dispatched = {}
        self._move2state:State = None
        self._quit = False
        self.dt = 0.01
        self.early_update = None

    def register_event(self, event_type, callback):
        if event_type not in self._events_suscribers:
            self._events_suscribers[event_type] = []
        self._events_suscribers[event_type].append(callback)

    def dispatch_event(self, event):
        self._events_dispatched[event.type] = event

    def _process_event(self):
        for event_type in self._events_dispatched:
            if event_type in self._events_suscribers:
                for callback in self._events_suscribers[event_type]:
                    callback(self._events_dispatched[event_type])

    def unregister_event(self, event_type, callback):
        if event_type in self._events_suscribers:
            self._events_suscribers[event_type].remove(callback)

    def _update(self):
        if self.early_update is not None:
            self.early_update()
        self._process_event()
        self._state_transfer()
        if self._state is not None:
            self._state.on_pre_update()
            self._state.on_update()
            self._state.on_post_update()


    def _state_transfer(self):
        if self._move2state is not None:
            print(f'[FiniteStateMachine] transfer state from {self._state} to {self._move2state}')
            self._state.on_exit()
            self._state._fsm = None
            self._state = self._move2state
            self._move2state = None
            self._state._fsm = self
            self._state.on_enter()

    def run(self):
        self._state._fsm = self
        self._state.on_enter()
        while not self._quit:
            self._update()
            time.sleep(self.dt)
        self._state.on_exit()
        self._state._fsm = None

    def quit(self):
        self._quit = True

    
    def move_to(self, state: State):
        move_to_state(self, state)

class Move2State:
    def __init__(self, fsm:FiniteStateMachine, state:State):
        self.fsm = fsm
        self.state = state
    
    def action(self):
        return move_to_state(self.fsm, self.state)

def move_to_state(fsm:FiniteStateMachine, state:State):
    fsm._move2state = state