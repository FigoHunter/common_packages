class DataCenter(dict):
    @classmethod
    def get_instance(cls):
        if not hasattr(cls,'__instance__'):
            cls.__instance__ = DataCenter()
        return cls.__instance__

    def setData(self, key, value):
        self[key] = value
    
    def getData(self, key, default=None):
        return self.get(key, default)