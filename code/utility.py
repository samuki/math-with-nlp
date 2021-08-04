def load_key(path):
    with open(path, 'r') as f: 
        key = f.readlines()[0]
    return key

def read_saxton_file(path):
    with open(path, 'r') as f: 
        lines = f.read().split('\n')
    print(lines)
    #return key