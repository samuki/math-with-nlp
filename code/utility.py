def load_key(path):
    with open(path, 'r') as f: 
        key = f.readlines()[0]
    return key