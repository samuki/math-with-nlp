import os
import openai
from GPTJ.Basic_api import SimpleCompletion
from GPTJ.gptj_api import Completion

def query_gpt3(key, prompt, engine='ada'):
    openai.api_key = key
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0,
        max_tokens=20,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response

def simple_query_gptj(prompt, max_length=100, temperature=0.01, top_probability=1.):
    query = SimpleCompletion(
        prompt, 
        length=max_length, 
        t=temperature, 
        top=top_probability
        )
    response = query.simple_completion()
    return response

def query_gptj(prompt, context_setting, context="Task", examples={}, User="Task", Bot="Calculator", \
    max_length=100, temperature=0.01, top_probability=1.):
    response = context_setting.completion(prompt,
              user=User,
              bot=Bot,
              max_tokens=max_length,
              temperature=temperature,
              top_p=top_probability)
    return response
