from . import environ
from . import file
from . import module
from . import path
from . import cache

def get_uuid():
    import uuid
    return uuid.uuid4().hex.replace('-','')