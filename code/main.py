from argparse import ArgumentParser
from query import query_gpt3, query_gptj
from utility import load_key, load_config
from exp import conduct_experiment


parser = ArgumentParser()
input_group = parser.add_argument_group('input_group')
input_group.add_argument('--config', dest='config', required=True, type=str)

def gpt3_example(): 
    path = 'keys/gpt3_key.txt'
    prompt = "hello\n"
    result = query_gpt3(load_key(path), prompt, engine='ada')

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

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    conduct_experiment(cfg)    

if __name__ =="__main__":
    main()