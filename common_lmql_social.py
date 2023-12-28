import pickle
import threading

from time import sleep
import timeit
import asyncio
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import json
# from common_generative import generate_prompt, model, tokenizer
from typing import List, Dict, Union
from collections import deque
from instagram.label_instagram import get_all_prompts
import numpy as np
import math
import fire
from tqdm import tqdm
import lmql


gpu_llms = {}
q_open_content = deque()
q_structured_content = deque()

LM_PATH = "/scratch/artahir/llama/hf_weights/llama2-13B-phase2-aug"
# LM_PATH = "/media/anique/Data/projects/llama-weights/llama2-7B-merged"
BATCH_SIZE = 15

# IMPORTANT escape the [,] and {,} characters for lmql
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:\n"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:\n"""


response_format = """
a) The post with the active ID ([POSTID]) [SUMMARY]
b) [BULLYING], the active post is [EXPSTART][BULLYINGEXTRA]
c) [ANTIBULLYING], the active post is [EXPSTART2][ANTIBULLYINGEXTRA]
d) The reason the post is [REASONBULL1] bullying is because [RSTART1][REASONBULL2]
e) The reason the post is [REASONANTIBULL1] anti-bullying is because [RSTART2][REASONANTIBULL2]
f) Whether the post is neither bullying or anti-bullying can be determined by considering answers to (b) and (c). Thus, [RSTART3][REASONNEI]
"""

PREDICATES = """
    len(POSTID) < 10 and 
    REGEX(POSTID, r"([0-9]|\))+") and 
    STOPS_BEFORE(POSTID, ")") and
    len(SUMMARY) < 100 and
    STOPS_BEFORE(SUMMARY, "b)") and
    BULLYING in ["Yes", "No"] and 
    len(BULLYINGEXTRA) < 100 and len(BULLYINGEXTRA) > 1 and  
    REGEX(EXPSTART, r"[a-z]{3}") and REGEX(EXPSTART2, r"[a-z]{3}") and
    STOPS_BEFORE(BULLYINGEXTRA, "c)") and
    ANTIBULLYING in ["Yes", "No"] and
    len(ANTIBULLYINGEXTRA) < 100 and 
    STOPS_BEFORE(ANTIBULLYINGEXTRA, "d)") and 
    REASONBULL1 in ["not considered", "considered"] and
    len(REASONBULL2) < 200 and REGEX(RSTART1, r"[a-z]{3}") and REGEX(RSTART2, r"[a-z]{3}") and REGEX(RSTART3, r"[a-z]{3}") and
    STOPS_BEFORE(REASONBULL2, "e)") and
    REASONANTIBULL1 in ["not considered", "considered"] and
    len(REASONANTIBULL2) < 200 and 
    STOPS_BEFORE(REASONANTIBULL2, "f)") and
    len(REASONNEI) < 200 and 
    STOPS_BEFORE(REASONNEI, ".")
"""


def extract_answers(resp):
    response_variables = resp[0].variables

    return {
        'postid': response_variables['POSTID'],
        'summary': response_variables['SUMMARY'],
        'bullying': response_variables['BULLYING'].strip(),
        'bullyingextra': (response_variables['EXPSTART'] + response_variables['BULLYINGEXTRA']).strip(),
        'antibullying': response_variables['ANTIBULLYING'].strip(),
        'antibullyingextra': (response_variables['EXPSTART2'] + response_variables['ANTIBULLYINGEXTRA']).strip(),
        'reasonbull1': response_variables['REASONBULL1'].strip(),
        'reasonbull2': (response_variables['RSTART1'] + response_variables['REASONBULL2']).strip(),
        'reasonantibull1': response_variables['REASONANTIBULL1'].strip(),
        'reasonantibull2': (response_variables['RSTART2'] + response_variables['REASONANTIBULL2']).strip(),
        'reasonnei2': (response_variables['RSTART3'] + response_variables['REASONNEI']).strip(),
    }


async def convert_response(prompt: str, predicates: str, decoder='argmax', n=2, model=LM_PATH):
    decoder_str = 'argmax' if decoder=='argmax' else f'beam({n})'
    # clean the prompt
    query_string = f'''{decoder_str}
"""{prompt}"""
where
{predicates}'''
    query_string = query_string.strip()
    try:
        query = lmql.query(query_string, model=model, decoder=decoder, n=n)
        response = await query()
    except Exception as e:
        print(e)
        response = None
    return prompt, response

async def _get_lmql_responses_batched(prompts, decoder='argmax', n=2, model=LM_PATH):
    batch_executions = []
    for prompt in prompts:
        prompt = prompt.replace("[", "")\
              .replace("]", "")\
              .replace("{", "")\
              .replace("}", "")\
              .replace('"""', "")
        prompt = prompt + response_format
        response_fn = convert_response(prompt, PREDICATES, decoder=decoder, n=n, model=model)
        batch_executions.append(response_fn)
    responses = await asyncio.gather(*batch_executions)
    return responses

def get_lmql_responses_batched(prompts, batch_size=10, decoder='argmax', n=2, model=LM_PATH):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        results.extend([None if r is None else (p, extract_answers(r)) for p, r in asyncio.run(_get_lmql_responses_batched(prompts[i:i+batch_size], decoder=decoder, n=n, model=model))])
    return results

def process_que(prompts, part=-1):
    for prompt in prompts:
        q_open_content.append(prompt + response_format)
    while len(q_structured_content) < len(prompts):
        sleep(1)

    if len(q_structured_content)>100 and len(q_structured_content) % 100 == 0:
        print(f"Processed {len(q_structured_content)} prompts")
        # save the results in a cache
        with open(f'cache_part_{part}.pkl', 'w') as response_file:
            pickle.dump(list(q_structured_content), response_file)

    return [q_structured_content.popleft() for _ in range(len(prompts))]

def get_instagram_responses(part: int, total_parts: int):
    instagram_prompts = get_all_prompts()
    # sort the prompts
    instagram_prompts = sorted(instagram_prompts)
    samples_per_part = math.ceil(len(instagram_prompts) / total_parts)

    # get the start and end index for this part
    start_index = (part - 1) * samples_per_part
    end_index = min(part * samples_per_part, len(instagram_prompts))

    # filter the prompts
    prompts = instagram_prompts[start_index:end_index]
    results = get_lmql_responses_batched(prompts, batch_size=BATCH_SIZE)

    # process the prompts
    # results = process_que(prompts, part=part)

    # save the results
    with open(f'instagram_responses_part_{part}.pkl', 'wb') as respones_file:
        pickle.dump(results, respones_file)

    return process_que(prompts)


if __name__ == "__main__":
    # serve in a seperate thread
    # threading.Thread(target=lmql.serve, args=(LM_PATH,), kwargs={'cuda': True, 'load_in_8bit': True}).start()
    # model = lmql.model(f"local:{LM_PATH}", cuda=True, load_in_8bit=True)

    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(convert_response("Question: What is an alpaca?\nAnswer: [ANSWER]", predicates="len(ANSWER) < 100", model='gpt2-medium'))
    fire.Fire(get_instagram_responses)


