import pandas as pd
from utility import load_data

def calc_accuracy(path, is_df=False):
    if is_df:
        results = load_data(path, is_df)
        return results.loc[results['Prediction'] == results['Ground truth']].shape[0]/\
            results.shape[0]
    else: 
        results = load_data(path)
        #print(results)
        results = pd.DataFrame.from_dict(results,orient='index')
        print(results.shape[0])
        return results.loc[results['Prediction'] == results['Ground truth']].shape[0]/\
            results.shape[0]
