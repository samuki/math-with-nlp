import pandas as pd
import os
import re
import numpy as np
from collections import Counter
from utility import load_data
from plot import plot_results, plot_grouped_lengths

def strip_end(text, suffix):
    if suffix and text.endswith(suffix):
        return text[:-len(suffix)]
    return text

def calc_accuracy(path, is_df=False):
    results=load_data(path, is_df)
    print(results.shape[0])
    return results.loc[results['Prediction'] == results['Ground truth']].shape[0]/\
        results.shape[0]

def get_mistakes(path, is_df=False): 
    results = load_data(path, is_df)
    mistakes = results.loc[results['Prediction'] != results['Ground truth']]
    return mistakes

def get_task_from_path(path):
    task = path.split('arithmetic__')[1].split('.')[0]
    task = strip_end(task, '_hard')
    task = strip_end(task, '_medium')
    return task


def print_group(group, n=10):
    df_dict = {'Task':[], 'Token':[], 'Frequency':[], 'Accuracy':[]}
    for task in group: 
        print(task)
        sorted_dict = sorted(group[task], key=lambda k: group[task][k]['all'], reverse=True)
        for token in sorted_dict[:n]:
            df_dict['Task'].append(task)
            df_dict['Token'].append(token)
            df_dict['Frequency'].append(str(group[task][token]['all']))
            df_dict['Accuracy'].append(str(round(group[task][token]['correct']/group[task][token]['all'], 2)))
            print(token, ' ', str(group[task][token]['all']), ' ', str(round(group[task][token]['correct']/group[task][token]['all'], 2)))
        print('################################')
    df = pd.DataFrame(data=df_dict)
    
    print(df.to_latex(columns=["Task", "Token", "Frequency", "Accuracy"], index=False))

def group_results_by_string_length(directory):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm
    group = {}
    for exp in os.listdir(directory):
        if not "generated" in exp:
            task = get_task_from_path(exp)
            exp = os.path.join(directory, exp)
            if not task in group: 
                group[task] = {'lengths':[], 'all':[]}
            df =load_data(exp)
            lengths = [len(sent) for sent in df['Question'].to_list()]
            df['Lengths'] = lengths
            all = np.zeros(df.shape[0])
            all[df['Prediction'] == df['Ground truth']]=1 
            group[task]['lengths'].extend(lengths)
            group[task]['all'].extend(list(all))
            
    return group

def group_results_by_string(directory):
    group = {}
    for exp in os.listdir(directory):
        if not "generated" in exp:
            task = get_task_from_path(exp)
            exp = os.path.join(directory, exp)
            if not task in group: 
                group[task] = {}
            df =load_data(exp)
            tokens = Counter([token  for sent in df['Question'].to_list() for token in sent.lower().split()])
            for token in sorted(tokens, key=lambda k: tokens[k], reverse=True)[:10]:
                wrong = df.loc[(df['Question'].str.contains(token, case=False, regex=False)) & (df['Prediction'] != df['Ground truth'])]
                correct = df.loc[(df['Question'].str.contains(token, case=False, regex=False)) & (df['Prediction'] == df['Ground truth'])]
                if not token in group[task]:
                    group[task][token] = {'correct':0, 'all':0}
                group[task][token]['correct'] += correct.shape[0]
                group[task][token]['all']+= wrong.shape[0]+correct.shape[0]
    return group

def get_difficulty_results_table(directory, difficulties=['easy', 'medium', 'hard']):
    task_dict = {}
    for exp in os.listdir(directory):
            exp = os.path.join('experiments', exp)
            task = exp.strip('arithmetic__').strip('_medium').strip('_hard')
            if not task in task_dict: 
                task_dict[task] = {diff:0 for diff in difficulties}

    table = ""
    for diff in difficulties:
        table += diff + "&" 
        for exp in os.listdir(directory):
            exp = os.path.join('experiments', exp)
            print(exp)
            acc = calc_accuracy(exp)
            ending = exp.strip('.pkl').split("_")[-1]
            #print(ending)
            ending = ending if ending in difficulties else 'easy'
            if ending == diff:
                table += str(round(acc, 2)) + '&'
        table += r'\\'
    print(table)

def main():
    path="experiments/arithmetic__add_or_sub"
    plot_dict = {'Task':[], "Accuracy":[]}
    for exp in os.listdir('experiments'):
        if '5' in exp: 
            plot_dict['Task'].append(" ".join(exp.strip('.pkl').split("_")))
            exp = os.path.join('experiments', exp)
            acc = calc_accuracy(exp)
            plot_dict["Accuracy"].append(acc)
            errors = get_mistakes(exp)
            print('Task ', exp)
            print("Acc ", acc)
        #if "mul" in exp and not "multiple" in exp: 
            #print('errors ', errors.head(10).to_latex(columns=["Question", "Prediction", "Ground truth"]))
    #group = group_results_by_string('experiments')
    #print_group(group)
    #group = group_results_by_string_length('experiments')
    #plot_grouped_lengths(group, save=True)
    #get_difficulty_results_table('experiments', difficulties=['easy', 'medium', 'hard'])

    #plot_results(plot_dict, save=True)

if __name__=="__main__":
    main()
