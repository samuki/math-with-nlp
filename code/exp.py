import openai
from GPTJ.gptj_api import Completion
import pandas as pd
import logging
from multiprocessing.dummy import Pool, Manager

# own library
from query import query_gptj, query_gptj_original_api, query_gpt3
from utility import make_logger, save_data, load_data
from data import read_saxton_file

global cfg 

def worker(i):
    question = list(dataset[i].keys())[0]
    #print(question)
    ground_truth = list(dataset[i].values())[0]
    question = context + '\n'+ question+ '\n'
    #response = query_gptj(prompt=question, context_setting=context_setting)
    if engine == 'gptj':
        response = query_gptj_original_api(prompt=question)
        response = response['text']
    elif engine.startswith('gpt3'):
        gpt3_engine = engine.split("_")[1]
        response = query_gpt3(prompt=question, engine=gpt3_engine)
        response = response['choices'][0]['text']
    #if response is not None:
    first_answer = response.split('\n')[0].strip("$")
    logger.info('Question number %8d : %s', i, question)
    logger.info('Prediciton: %s', first_answer)
    logger.info('Ground truth %s', ground_truth)
    logger.info('The result is %r', first_answer==ground_truth)
    logger.info('\n')
    results[i]= {'Question':question, 'Prediction':first_answer, 'Ground truth':ground_truth}
    save_data(results, out_path)

def conduct_experiment(cfg):
    global dataset, examples, context_setting, in_path, out_path, results,\
         context, logger, engine
    log_string = "{}.log".format('logs/'+cfg['task'])
    logger = make_logger(log_string)
    in_path, out_path = cfg['in_path']+cfg['name']+'.txt', cfg['out_path']+cfg['task']
    manager = Manager()
    results = manager.dict()
    dataset = read_saxton_file(in_path)
    examples = {d:dataset[i][d].strip('.') for i in range(cfg['example'][0], cfg['example'][1])\
         for d in dataset[i]}
    context =  "\n".join([key+"\n"+examples[key] for key in examples])
    context_setting = Completion(context, examples)
    pool_size = cfg['pool_size']
    pool = Pool(processes=pool_size)
    cfg['end'] = max(dataset, key=int) if not cfg['end'] else cfg['end']
    engine = "gptj"
    if 'engine' in cfg:
        engine = cfg['engine']
    logger.info("Loaded parameters")
    pool.map(worker, range(cfg['start'],cfg['end']))
    pool.close()
    pool.join()
