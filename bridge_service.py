from bridge_service_client import BridgeServiceClient
from bridge_service_client.utils import set_env_var, get_env_var

_client = None

def set_client(name=None, addr=None, port=None):
    if isinstance(port, str):
        port = int(port)
    global _client
    if _client is None:
        if name is None or not name:
            name = 'blender'
        _client = BridgeServiceClient(name, addr, port)
    else:
        print(f'[Warning] Client already exists with name {_client.name}')

def get_client(name=None, addr=None, port=None, start_if_not_exists=True):
    global _client
    if _client is None:
        if start_if_not_exists:
            set_client(name, addr, port)
    else:
        if name is not None and _client.name != name:
            print(f'[Warning] Client name {name} does not match existing client name {_client.name}')
        if addr is not None and _client.address != addr:
            print(f'[Warning] Client address {addr} does not match existing client address {_client.address}')
        if port is not None and _client.port != port:
            print(f'[Warning] Client port {port} does not match existing client port {_client.port}')
    return _client

def connect():
    get_client().send_clear()
    get_client().connect(True)

def action(key=None):
    return get_client().action(key)
