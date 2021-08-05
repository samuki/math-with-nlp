import openai
from GPTJ.gptj_api import Completion
import pandas as pd
import logging
from multiprocessing.dummy import Pool, Manager

# own library
from query import query_gptj 
from utility import make_logger, save_data, load_data
from data import read_saxton_file

global cfg 

def worker(i):
    question = list(dataset[i].keys())[0]
    ground_truth = list(dataset[i].values())[0]
    response = query_gptj(prompt=question, context_setting=context_setting)
    first_answer = response.split('\n')[0]
    logger.info('Question number %8d : %s', i, question)
    logger.info('Prediciton: %s', first_answer)
    logger.info('Ground truth %s', ground_truth)
    logger.info('The result is %r', first_answer==ground_truth)
    logger.info('\n')
    results[i]= {'Question':question, 'Prediction':first_answer, 'Ground truth':ground_truth}
    save_data(results, out_path)

def conduct_experiment(cfg):
    global dataset, examples, context_setting, in_path, out_path, results,\
         context, logger
    log_string = "{}.log".format('logs/'+cfg['task'])
    logger = make_logger(log_string+"logger")
    in_path, out_path = cfg['in_path']+cfg['task']+'.txt', cfg['out_path']+cfg['task']
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
    logger.info("Loaded parameters")
    pool.map(worker, range(cfg['start'],cfg['end']))
    pool.close()
    pool.join()
