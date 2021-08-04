from query import query_gpt3, query_gptj, simple_query_gptj
from utility import load_key, read_saxton_file
import openai

def main():
    path = 'keys/gpt3_key.txt'
    prompt = "hello\n"
    #result = query_gpt3(load_key(path), prompt, engine='ada')
    #print(result)
    #print(read_saxton_file('data/arithmetic__add_or_sub.txt'))
    #print(simple_query_gptj(prompt="def perfect_square(num):"))
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

if __name__ =="__main__":
    main()