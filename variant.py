
class VariantManager:
    def __init__(self):
        self._version_map = {}

    def variant(self, variant):
        def decorator(func):
            self._version_map[variant] = func
            return func
        return decorator

    def get_variant(self, variant):
        return self._version_map.get(variant, None)
