import openai
from GPTJ.gptj_api import Completion
import pandas as pd

# own library
from query import query_gpt3, query_gptj, simple_query_gptj
from utility import load_key, save_data, load_data
from data import read_saxton_file

def gptj_calculation_dummy_test():
    context = "This is a calculator bot that will answer basic math questions"
    examples = {
        "5 + 5": "10",
        "6 - 2": "4",
        "4 * 15": "60",
        "10 / 5": "2",
        "144 / 24": "6",
        "7 + 1": "8"
    }
    prompt = "48 / 6"
    print(query_gptj(prompt=prompt, examples=examples, context=context))

def gpt3_example(): 
    path = 'keys/gpt3_key.txt'
    prompt = "hello\n"
    result = query_gpt3(load_key(path), prompt, engine='ada')

def test_saxton_example(in_path, out_path, ex=[0,10], prompt=11):
    dataset = read_saxton_file(in_path)
    examples = {d:dataset[i][d].strip('.') for i in range(ex[0], ex[1]) for d in dataset[i]}
    context =  "\n".join([key+"\n"+examples[key] for key in examples])
    context_setting = Completion(context, examples)
    results = pd.DataFrame(columns=['Question', 'Prediction', 'Ground Truth'])
    for i in range(prompt,max(dataset, key=int)):
        question = list(dataset[i].keys())[0]
        ground_truth = list(dataset[i].values())[0]
        response = query_gptj(prompt=question, context_setting=context_setting, \
            examples=examples, context=context, max_length=20)
        first_answer = response.split('\n')[0]
        results.loc[i] = [question, first_answer, ground_truth]
        print('Question number ',i, ' : ', question)
        print('Prediciton: ', first_answer)
        print('Ground truth ', ground_truth)
        print(first_answer==ground_truth)
        print('\n')
        save_data(results, out_path, is_df=True)
        results = load_data(out_path, is_df=True)
    return results

def calc_accuracy(path, is_df=True):
    results = load_data(path, is_df)
    return results.loc[results['Prediction'] == results['Ground Truth']].shape[0]/\
        results.shape[0]

def main():
    task = 'arithmetic__add_or_sub'
    in_path = 'data/mathematics_dataset-v1.0/train-easy/'+task+'.txt'
    out_path = 'experiments/'+task
    test_saxton_example(in_path=in_path,out_path=out_path)
    print(calc_accuracy(out_path))
    

if __name__ =="__main__":
    main()