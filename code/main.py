from query import query_gpt3
from utility import load_key
import openai

def main():
    path = 'keys/gpt3_key.txt'
    prompt = "hello\n"
    result = query_gpt3(load_key(path), prompt, engine='ada')
    print(result)

if __name__ =="__main__":
    main()