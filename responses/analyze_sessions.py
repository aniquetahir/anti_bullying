from instagram.label_instagram import get_prompts, get_all_prompts
import os
import pickle
import math
from io import StringIO
import pandas as pd


RESPONSE_FOLDER = 'responses/13B_phase2'

def get_part_prompts_and_sessions(part, total):
    prompts, sessions = get_all_prompts()

    instagram_prompts = sorted(prompts)
    instagram_sessions = sorted(zip(prompts, sessions), key=lambda x: x[0])
    instagram_sessions = [session for _, session in instagram_sessions]

    samples_per_part = math.ceil(len(instagram_prompts) / total)

    # get the start and end index for this part
    start_index = (part - 1) * samples_per_part
    end_index = min(part * samples_per_part, len(instagram_prompts))

    # filter the prompts
    prompts = instagram_prompts[start_index:end_index]
    sessions = instagram_sessions[start_index:end_index]

    return prompts, sessions



def map_responses_to_sessions(response_folder, num_parts):
    files_list = os.listdir(response_folder)
    # get the parts that are available
    part_responses = {}
    for file_name in files_list:
        if 'part_' in file_name:
            part_num = int(file_name.split('part_')[1].split('.')[0])
            with open(os.path.join(response_folder, file_name), 'rb') as f:
                part_responses[part_num] = pickle.load(f)

    # get the sessions and prompts for each reply
    for part in part_responses.keys():
        prompts, sessions = get_part_prompts_and_sessions(part, num_parts)
        print(f"Part {part} has {len(prompts)} prompts and {len(part_responses[part])} responses")
        if len(prompts) != len(part_responses[part]):
            print("ERROR: number of prompts and responses don't match")
            continue
        part_responses[part] = list(zip(prompts, sessions, part_responses[part]))

if __name__ == "__main__":
    map_responses_to_sessions(RESPONSE_FOLDER, 10)