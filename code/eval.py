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

def main():
    path="experiments/arithmetic__add_or_sub"
    print(calc_accuracy(path))

if __name__=="__main__":
    main()
