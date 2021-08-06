import pandas as pd
import os
from utility import load_data

def calc_accuracy(path, is_df=False):
    results=load_data(path, is_df)
    print(results.shape[0])
    return results.loc[results['Prediction'] == results['Ground truth']].shape[0]/\
        results.shape[0]

def get_mistakes(path, is_df=False): 
    results = load_data(path, is_df)
    mistakes = results.loc[results['Prediction'] != results['Ground truth']]
    return mistakes

def main():
    path="experiments/arithmetic__add_or_sub"
    for exp in os.listdir('experiments'):
        exp = os.path.join('experiments', exp)
        acc = calc_accuracy(exp)
        errors = get_mistakes(exp)
        print('Task ', exp)
        print("Acc ", acc)
        print('errors ', errors.head(5))

if __name__=="__main__":
    main()
