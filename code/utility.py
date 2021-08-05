import pickle
import pandas as pd 

def load_key(path):
    with open(path, 'r') as f: 
        key = f.readlines()[0]
    return key

def save_data(data, path, is_df=False):
    if is_df:
        data.to_pickle(path+'.pkl')
    else:
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_data(path, is_df=False):
    if is_df:
        return pd.read_pickle(path+'.pkl')
    else:
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)