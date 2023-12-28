import pickle
import threading

from time import sleep
import timeit
import asyncio
from typing import List, Dict, Union
from collections import deque
import numpy as np
import math
import fire
from tqdm import tqdm
import lmql
import json
from common_generative import generate_prompt

SENTIMENT_DATA_PATH = "sentiment_alpaca.json"
STANCE_DATA_PATH = "stance_alpaca.json"

"""
Example LMQL input:
beam(2)
\"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The input contains a tweet from social media.
First explain the tweet. Then, select the sentiment of the tweet given in the input. Select the sentiment from:
- negative
- neutral
- positive

### Input
So disappointed in wwe summerslam! I want to see john cena wins his 16th title

### Response:
Explanation:
[EXPLN]

Sentiment:
[RESPONSE]
\"\"\"
from 
    lmql.model("/scratch/artahir/llama/hf_weights/llama2-13B-phase2-aug")
where
    len(EXPLN) < 200 and
    len(RESPONSE) < 100 and 
    RESPONSE in ["negative", "neutral", "positive"]
"""

INSTRUCTION_SENTIMENT_COT = """The input contains a tweet from social media.
First explain the tweet. Then, select the sentiment of the tweet given in the input. Select the sentiment from:
- negative
- neutral
- positive"""

INSTRUCTION_SENTIMENT = """The input contains a tweet from social media.
Select the sentiment of the tweet given in the input. Select the sentiment from:
- negative
- neutral
- positive"""

INSTRUCTION_STANCE_COT = """The input contains a tweet from social media.
First explain the tweet. Then, select the stance of the tweet given in the input. Select the stance from:
- none
- against
- favor"""

INSTRUCTION_STANCE = """The input contains a tweet from social media.
Select the stance of the tweet given in the input. Select the stance from:
- none
- against
- favor"""

RESPONSE_FORMAT_STANCE_SENTIMENT = """
[RESPONSE]
""".strip()

RESPONSE_FORMAT_STANCE_SENTIMENT_COT = """
Explanation: 
[EXPLN]

Label: 
[RESPONSE]
""".strip()

# TODO figure out the best ones
PREDICATES_SENT = """
len(RESPONSE) < 100 and
RESPONSE in ["negative", "neutral", "positive"]
"""

# TODO complete the template for best predicates
PREDICATES_SENT_COT = """
len(EXPLN) < 200 and
STOPS_BEFORE(EXPLN, "\\n") and
len(RESPONSE) < 100 and
RESPONSE in ["negative", "neutral", "positive"]
"""

PREDICATES_STANCE = """
len(RESPONSE) < 100 and
RESPONSE in ["none", "against", "favor"]
"""

PREDICATES_STANCE_COT = """
len(EXPLN) < 200 and
STOPS_BEFORE(EXPLN, "\\n") and
len(RESPONSE) < 100 and
RESPONSE in ["none", "against", "favor"]
"""

dict_choices = {
    ('stance', 'standard'): {
        'instruction': INSTRUCTION_STANCE,
        'response_format': RESPONSE_FORMAT_STANCE_SENTIMENT,
        'predicates': PREDICATES_STANCE,
    },
    ('stance', 'cot'): {
        'instruction': INSTRUCTION_STANCE_COT,
        'response_format': RESPONSE_FORMAT_STANCE_SENTIMENT_COT,
        'predicates': PREDICATES_STANCE_COT,
    },
    ('sentiment', 'standard'): {
        'instruction': INSTRUCTION_SENTIMENT,
        'response_format': RESPONSE_FORMAT_STANCE_SENTIMENT,
        'predicates': PREDICATES_SENT,
    },
    ('sentiment', 'cot'): {
        'instruction': INSTRUCTION_SENTIMENT_COT,
        'response_format': RESPONSE_FORMAT_STANCE_SENTIMENT_COT,
        'predicates': PREDICATES_SENT_COT,
    }
}


gpu_llms = {}
q_open_content = deque()
q_structured_content = deque()

# LM_PATH = "/scratch/artahir/llama/hf_weights/llama2-13B-phase2-aug"
LM_PATH = "/media/anique/Data/projects/llama-weights/llama2-7B-merged"
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


# PREDICATES = """
#     len(POSTID) < 10 and
#     REGEX(POSTID, r"([0-9]|\))+") and
#     STOPS_BEFORE(POSTID, ")") and
#     len(SUMMARY) < 100 and
#     STOPS_BEFORE(SUMMARY, "b)") and
#     BULLYING in ["Yes", "No"] and
#     len(BULLYINGEXTRA) < 100 and len(BULLYINGEXTRA) > 1 and
#     REGEX(EXPSTART, r"[a-z]{3}") and REGEX(EXPSTART2, r"[a-z]{3}") and
#     STOPS_BEFORE(BULLYINGEXTRA, "c)") and
#     ANTIBULLYING in ["Yes", "No"] and
#     len(ANTIBULLYINGEXTRA) < 100 and
#     STOPS_BEFORE(ANTIBULLYINGEXTRA, "d)") and
#     REASONBULL1 in ["not considered", "considered"] and
#     len(REASONBULL2) < 200 and REGEX(RSTART1, r"[a-z]{3}") and REGEX(RSTART2, r"[a-z]{3}") and REGEX(RSTART3, r"[a-z]{3}") and
#     STOPS_BEFORE(REASONBULL2, "e)") and
#     REASONANTIBULL1 in ["not considered", "considered"] and
#     len(REASONANTIBULL2) < 200 and
#     STOPS_BEFORE(REASONANTIBULL2, "f)") and
#     len(REASONNEI) < 200 and
#     STOPS_BEFORE(REASONNEI, ".")
# """




def extract_answers(resp):
    response_variables = resp[0].variables
    return response_variables


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

async def _get_lmql_responses_batched(prompts, decoder='argmax', n=2, model=LM_PATH, task='stance', mode='standard'):
    """

    :param prompts:
    :param decoder:
    :param n:
    :param model: path to the language model parameters
    :param task: 'stance' or 'sentiment'
    :param mode: 'standard' or 'cot'
    :return:
    """
    batch_executions = []
    template_items = dict_choices[(task, mode)]
    instruction = template_items['instruction']
    response_format = template_items['response_format']
    predicates = template_items['predicates']
    for prompt in prompts:
        input = prompt['input']
        input = input.replace("[", "")\
              .replace("]", "")\
              .replace("{", "")\
              .replace("}", "")\
              .replace('"""', "")
        prompt_str = generate_prompt(instruction, input=input)
        prompt_str += response_format
        response_fn = convert_response(prompt_str, predicates, decoder=decoder, n=n, model=model)
        batch_executions.append(response_fn)
    responses = await asyncio.gather(*batch_executions)
    return responses

def get_lmql_responses_batched(prompts, batch_size=10, decoder='argmax', n=2, model=LM_PATH, task='stance', mode='standard'):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        results.extend([None if r is None else (p, extract_answers(r)) for p, r in
                        asyncio.run(_get_lmql_responses_batched(prompts[i:i+batch_size],
                                    decoder=decoder, n=n, model=model, task=task, mode=mode))])
    return results

def get_sentiment_responses(part, total_parts, task='sentiment', mode='standard'):
    # load the data from the file
    with open(SENTIMENT_DATA_PATH, 'r') as sentiment_file:
        sentiment_data = json.load(sentiment_file)

    sentiment_data = sentiment_data['test']

    # sort the data in terms of the input
    sentiment_data = sorted(sentiment_data, key=lambda x: x['input'])
    samples_per_part = math.ceil(len(sentiment_data) / total_parts)

    # get the samples for this part
    start_idx = (part - 1) * samples_per_part
    end_idx = min(part * samples_per_part, len(sentiment_data))

    # filter the prompts
    sentiment_data = sentiment_data[start_idx:end_idx]

    results = get_lmql_responses_batched(sentiment_data, batch_size=BATCH_SIZE, task='sentiment', mode='standard')

    # save the results
    with open(f"sentiment_results_{part}_of_{total_parts}_{task}_{mode}.pkl", 'wb') as sentiment_file:
        pickle.dump(results, sentiment_file)
    # print(len(results))


if __name__ == "__main__":
    # serve in a seperate thread
    # threading.Thread(target=lmql.serve, args=(LM_PATH,), kwargs={'cuda': True, 'load_in_8bit': True}).start()
    # model = lmql.model(f"local:{LM_PATH}", cuda=True, load_in_8bit=True)
    # get_sentiment_responses()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(convert_response("Question: What is an alpaca?\nAnswer: [ANSWER]", predicates="len(ANSWER) < 100", model='gpt2-medium'))
    # fire.Fire(get_instagram_responses)
    fire.Fire(get_sentiment_responses)


