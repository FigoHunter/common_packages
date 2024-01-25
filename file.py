import os
def writeToFile(path, content=""):
    path = os.path.abspath(path)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path,'w') as f:
        f.write(content)

def readFromFile(path):
    with open(path, 'r') as f:
        data = f.read()
    return data

def prepareDir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def copyTo(from_path, to_path, overwrite=False):
    from_path = os.path.abspath(from_path)
    to_path = os.path.abspath(to_path)
    dir = os.path.dirname(to_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if os.path.exists(to_path):
        if overwrite:
            os.remove(to_path)
        else:
            raise Exception(f'File exists: {to_path}')
    import shutil
    shutil.copy(from_path, to_path)