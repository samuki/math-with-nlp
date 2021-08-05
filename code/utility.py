import pickle
import pandas as pd 
import yaml 
import logging

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

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def make_logger(log_file: str = None, name='logger'):
    """
    Create a logger for logging the training/testing process.

    :param log_file: path to file where log is stored as well
    :return: logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    return logger