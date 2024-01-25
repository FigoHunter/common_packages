def get_uuid():
    import uuid
    return uuid.uuid4().hex.replace('-','')