class FuncVariant:
    def __init__(self):
        self._variant_map = {}

    def variant(self, variant_name):
        def decorator(func):
            self._variant_map[variant_name] = func
            return func
        return decorator

    def get_func(self, variant_name):
        return self._variant_map.get(variant_name, None)
