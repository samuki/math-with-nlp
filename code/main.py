from query import query_gpt3, query_gptj, simple_query_gptj
from utility import load_key
from data import read_saxton_file
import openai

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

def test_saxton_example(path, ex=[0,10], prompt=11):
    context = "This is a calculator bot that will answer basic math questions"
    dataset = read_saxton_file(path)
    examples = {d:dataset[i][d] for i in range(ex[0], ex[1]) for d in dataset[i]}
    examples = {
        "5 + 5": "10",
        "6 - 2": "4",
        "4 * 15": "60",
        "10 / 5": "2",
        "144 / 24": "6",
        "7 + 1": "8"
    }
    print(examples)
    for i in range(prompt,30):
        question = list(dataset[i].keys())[0]
        soln = list(dataset[i].values())[0]
        print('Question: ', question)
        print('Prediciton: ', query_gptj(prompt=question, examples=examples, context=context, max_length=30))
        print('Ground truth ', soln)

def main():
    path = 'data/mathematics_dataset-v1.0/train-easy/arithmetic__add_or_sub.txt'
    test_saxton_example(path)
    

if __name__ =="__main__":
    main()