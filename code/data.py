def read_saxton_file(path):
    qa = {}
    with open(path, 'r') as f: 
        read=f.read()
        split = read.split('\n')
    pairs = [{split[i]:split[i+1]} for i in range(0,len(split)-1, 2)]
    for i, pair in enumerate(pairs):
        qa[i] = pair
    return qa
