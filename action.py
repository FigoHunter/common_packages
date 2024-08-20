class Action:
    def __init__(self):
        self._actions = []
        self._args = {}

    def register(self, callback, **kwargs):
        """Register an action"""
        self._actions.append(callback)
        self._args[callback] = kwargs

    def __iadd__(self, action):
        """Support for the += operator to add an action"""
        self._actions.append(action)
        return self

    def __isub__(self, action):
        """Support for the -= operator to remove an action"""
        self._actions.remove(action)
        if action in self._args:
            del self._args[action]
        return self

    def trigger(self, *args, **kwargs):
        """Invoke all registered actions"""
        for action in self._actions:
            action_args = self._args.get(action, {})
            action_args = {**kwargs, **action_args}
            action(*args, **action_args)
    
    def set_args(self, callback, **kwargs):
        """Set the arguments for the callback"""
        self._args[callback] = kwargs
