import pandas as pd
import numpy as np
from scipy import stats as st
import json
import os
import unicodedata
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
# from fa2 import ForceAtlas2
from functools import reduce, partial
from common import get_instagram_sessions, create_session_table
from common_generative import get_summary, get_paraphrase, get_summary_batched, get_paraphrase_batched
from augmentation import get_augmentation_questions, create_name_of_author_dataset, who_said_that, post_which_said, summarize_posts_by_author
from tqdm import tqdm
from collections import deque

summary_needed_list = []
paraphrase_needed_list = []

summaries_generated = {}
paraphrases_generated = {}
# summary_fn = get_summary
# paraphrase_fn = get_paraphrase

batch_size = 7

def summary_gatherer_fn(text):
    summary_needed_list.append(text)
    return f'<summary>{text}</summary>'

def paraphrase_gatherer_fn(text):
    paraphrase_needed_list.append(text)
    return f'<paraphrase>{text}</paraphrase>'


def get_session_augmentations_author_name(session, window_size=10):
    instructions = []
    x = get_augmentation_questions(create_name_of_author_dataset, session, window_size=window_size)
    for input, qa_pairs in x:
        # create the prompt
        for qa_pair in qa_pairs:
            post_id = qa_pair['post']
            instruction = f"What is the name of the author for the post with ID {post_id}?"
            response = qa_pair['author']
            instructions.append({
                'instruction': instruction,
                'input': input,
                'output': response
            })
    return instructions

def get_session_augmentations_who_said_that(session, window_size=10):
    instructions = []
    wst = partial(who_said_that, summary_fn=summary_fn)
    x = get_augmentation_questions(wst, session, window_size=window_size)
    for input, qa_pairs in x:
        # create the prompt
        for qa_pair in qa_pairs:
            summary = qa_pair['summary']
            instruction = f"Who is the author who has the following views?\n```\n{summary}\n```"
            response = qa_pair['author']
            instructions.append({
                'instruction': instruction,
                'input': input,
                'output': response
            })
    return instructions

def get_session_augmentations_summarize_posts_by_author(session, window_size=10):
    instructions = []
    spba = partial(summarize_posts_by_author, summary_fn=summary_fn)
    x = get_augmentation_questions(spba, session, window_size=window_size)
    for input, qa_pairs in x:
        # create the prompt
        for qa_pair in qa_pairs:
            author = qa_pair['author']
            instruction = f"Give a summary of the posts by the author {author}."
            response = qa_pair['summary']
            instructions.append({
                'instruction': instruction,
                'input': input,
                'output': response
            })
    return instructions

def get_session_augmentations_post_which_said(session, window_size=10):
    instructions = []
    pws = partial(post_which_said, paraphrase_fn=paraphrase_fn)
    x = get_augmentation_questions(pws, session, window_size=window_size)
    for input, qa_pairs in x:
        # create the prompt
        for qa_pair in qa_pairs:
            paraphrase = qa_pair['paraphrase']
            instruction = f"What is the ID of the post which said the following:\n```\n{paraphrase}\n```"
            response = qa_pair['post_id']
            instructions.append({
                'instruction': instruction,
                'input': input,
                'output': str(response)
            })
    return instructions

def replace_tag_template(tag, summary_dict, text):
    tag_begin = f'<{tag}>'
    tag_end = f'</{tag}>'
    if tag_begin in text:
        sum_key = text.split(tag_begin)[1].split(tag_end)[0]
        pre = text.split(tag_begin)[0]
        post = text.split(tag_end)[1]
        if sum_key in summary_dict.keys():
            text = pre + summary_dict[sum_key] + post
        else:
            text = None
    return text

if __name__ == "__main__":
    alpaca_instructions = []
    sessions = get_instagram_sessions()
    summary_fn = summary_gatherer_fn
    paraphrase_fn = paraphrase_gatherer_fn

    for session in tqdm(sessions):
        alpaca_instructions.extend(get_session_augmentations_who_said_that(session, window_size=10))
        alpaca_instructions.extend(get_session_augmentations_summarize_posts_by_author(session, window_size=10))
        alpaca_instructions.extend(get_session_augmentations_post_which_said(session, window_size=10))
        alpaca_instructions.extend(get_session_augmentations_author_name(session, window_size=10))

    print('augmentation template generated')
    print('summarizing in batches')
    summary_needed_list = list(set(summary_needed_list))
    paraphrase_needed_list = list(set(paraphrase_needed_list))


    if os.path.exists('instagram_summaries.json'):
        with open('instagram_summaries.json', 'r') as summary_file:
            summaries_generated = json.load(summary_file)
    if os.path.exists('instagram_paraphrases.json'):
        with open('instagram_paraphrases.json', 'r') as paraphrase_file:
            paraphrases_generated = json.load(paraphrase_file)
    if not os.path.exists('instagram_summaries.json'):
        for i in tqdm(range(0, len(summary_needed_list), batch_size)):
            summaries = get_summary_batched(summary_needed_list[i:i+batch_size])
            for j, s in enumerate(summary_needed_list[i:i+batch_size]):
                summaries_generated[s] = summaries[j]
        with open("instagram_summaries.json", 'w') as summary_file:
            json.dump(summaries_generated, summary_file, indent=2)
    print('paraphrasing in batches')

    if not os.path.exists('instagram_paraphrases.json'):
        for i in tqdm(range(0, len(paraphrase_needed_list), batch_size)):
            paraphrases = get_paraphrase_batched(paraphrase_needed_list[i:i+batch_size])
            for j, s in enumerate(paraphrase_needed_list[i:i+batch_size]):
                paraphrases_generated[s] = paraphrases[j]
        # Save the paraphrases

        with open("instagram_paraphrases.json", 'w') as paraphrase_file:
            json.dump(paraphrases_generated, paraphrase_file, indent=2)

    print('replacing strings in template')
    augmented_samples = []
    for i, sample in enumerate(tqdm(alpaca_instructions)):
        # extract the text to summarize
        instruction = sample['instruction']
        output = sample['output']
        # if 'paraphrase' in instruction or 'paraphrase' in output:
        #     print('paraphrase detected')
        try:
            instruction = replace_tag_template('summary', summaries_generated, instruction)
            instruction = replace_tag_template('paraphrase', paraphrases_generated, instruction)
            output = replace_tag_template('summary', summaries_generated, output)
            output = replace_tag_template('paraphrase', paraphrases_generated, output)
        except:
            continue

        if instruction is None or output is None:
            continue
        else:
            augmented_samples.append({
                'instruction': instruction,
                'input': sample['input'],
                'output': output
            })
    print('saving augmented samples')
    print('number of augmented samples:', len(augmented_samples))
    with open("instagram_alpaca.json", 'w') as alpaca_file:
        json.dump(augmented_samples, alpaca_file, indent=2)