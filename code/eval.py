import pandas as pd
from utility import load_data

def calc_accuracy(path, is_df=False):
    results=load_data(path, is_df)
    print(results.shape[0])
    return results.loc[results['Prediction'] == results['Ground truth']].shape[0]/\
        results.shape[0]

def main():
    path="experiments/arithmetic__add_or_sub"
    print(calc_accuracy(path))

if __name__=="__main__":
    main()
