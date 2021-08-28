from argparse import ArgumentParser
from query import query_gpt3, query_gptj, query_gptj_original_api
from utility import load_key, load_config
from exp import conduct_experiment
from GPTJ.gptj_api import Completion


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
    context =  "\n".join([key+"\n"+examples[key] for key in examples])
    #context_setting = Completion("", examples)
    #print(query_gptj(prompt=prompt, context_setting=context_setting))
    #print(query_gptj_original_api(context + '\n'+ +prompt+ '\n'))

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    conduct_experiment(cfg)
    #query_original_api(prompt)
    #gptj_calculation_dummy_test()

if __name__ =="__main__":
    main()