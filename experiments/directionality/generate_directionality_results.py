import json
import os
import pickle
from common_lmql_directionality import generate_prompt
import pandas as pd
from io import StringIO


def find_prompt_data(prompt, all_data):
    for x in all_data:
        if x['prompt'] == prompt.split('### Response:')[0] + '### Response:\n':
            # extract ground truth postid from csv
            csv = x['input']
            csv = StringIO(csv)
            df_input = pd.read_csv(csv)
            active_post_id = df_input.iloc[-1]['post_id']
            x['active_post_id'] = active_post_id
            return x
    return None

def match_prompts_to_ground_truth(raw_data, prompts):
    """
    returns ((gt_postid, gt_reply_to), (pred_postid, pred_reply_to))
    """
    # with open('experiments/directionality/4chan_directionality_inputs.json') as directionality_file:
    #     directionality_data = json.load(directionality_file)

    # create a prompt for each input in directionality data
    # prompts = [{**x, 'prompt': generate_prompt(x['input'])} for x in directionality_data if x['reply_to'] != '']
    raw_mapped_data = [(x[1], find_prompt_data(x[0], prompts)) for x in raw_data if x is not None]
    raw_mapped_data = [(x[0], x[1]) for x in raw_mapped_data if x[1] is not None]

    return raw_mapped_data


def calculate_statistics(matched_data):
    active_post_found = [int(x[0]['postid']) == int(x[1]['active_post_id']) for x in matched_data]
    reply_to_found = [int(x[0]['reply_to']) in [int(y) for y in x[1]['reply_to'].split(';') if y != ''] for x in matched_data]

    # p(reply_to_found | active_post_found)
    # filter out the ones where active post was not found
    condition_matched = [x for i, x in enumerate(matched_data) if active_post_found[i]]
    condition_reply_to_found = [int(x[0]['reply_to']) in [int(y) for y in x[1]['reply_to'].split(';') if y != ''] for x in condition_matched]

    # accuracy of active_post_found
    active_post_accuracy = sum(active_post_found) / len(active_post_found)
    # accuracy of reply_to_found
    reply_to_accuracy = sum(reply_to_found) / len(reply_to_found)
    # accuracy of reply_to_found given active_post_found
    reply_to_accuracy_given_active_post = sum(condition_reply_to_found) / len(condition_reply_to_found)

    return {
        'target post identified': active_post_accuracy, 
        'reply post identified': reply_to_accuracy, 
        'p(reply identified | target identified)': reply_to_accuracy_given_active_post
    }


def generate_results_table():
    results = {
        '7B': [],
        '13B': [],
        '13B-PEFT': [],
        '7B-PEFT': []
    }

    with open('experiments/4chan_directionality_responses_part_llama2-7B_1_of_1.pkl', 'rb') as f:
        raw_7B = pickle.load(f)
    
    with open('experiments/4chan_directionality_responses_part_llama2-13B_1_of_1.pkl', 'rb') as f:
        raw_13B = pickle.load(f)

    with open('experiments/4chan_directionality_responses_part_llama2-13B-phase2_1_of_1.pkl', 'rb') as f:
        raw_13B_PEFT = pickle.load(f)

    with open('experiments/4chan_directionality_responses_part_llama2-7B-merged_1_of_1.pkl', 'rb') as f:
        raw_7B_PEFT = pickle.load(f)

    """
    Raw data format:
    [
        [prompt, {'postid': ...,  'reply_to': ...}]
    ]
    """
    matched_7B = match_prompts_to_ground_truth(raw_7B[0], raw_7B[1])
    # calculate_statistics(matched_7B)
    matched_13B = match_prompts_to_ground_truth(raw_13B[0], raw_13B[1])
    # calculate_statistics(matched_13B)
    matched_13B_PEFT = match_prompts_to_ground_truth(raw_13B_PEFT[0], raw_13B_PEFT[1])
    # calculate_statistics(matched_13B_PEFT)
    matched_7B_PEFT = match_prompts_to_ground_truth(raw_7B_PEFT[0], raw_7B_PEFT[1])
    # calculate_statistics(matched_7B_PEFT)

    # create a csv of results
    results = {
        '7B': calculate_statistics(matched_7B),
        '13B': calculate_statistics(matched_13B),
        '13B-PEFT': calculate_statistics(matched_13B_PEFT),
        '7B-PEFT': calculate_statistics(matched_7B_PEFT)
    }

    df = pd.DataFrame(results)
    df.to_csv('experiments/directionality/directionality_results.csv')

    
    print(len(raw_7B))

    pass

if __name__ == "__main__":
    generate_results_table()