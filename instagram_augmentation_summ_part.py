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
            instruction = f"What is the name of the author for the post with ID: {post_id}?"
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
            instruction = f"Who is the author who has the following views:\n {summary}?"
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
            instruction = f"What is the ID of the post which said the following:\n {paraphrase}?"
            response = qa_pair['post_id']
            instructions.append({
                'instruction': instruction,
                'input': input,
                'output': response
            })
    return instructions


def get_summaries_partial(text_list, part, total):
    start = int(part - 1 * len(text_list) / total)
    end = int((part) * len(text_list) / total)
    text_to_summarize = text_list[start:end]
    summary_partial = {}
    for i in tqdm(range(0, len(text_to_summarize), batch_size)):
        summaries = get_summary_batched(text_to_summarize[i:i+batch_size])
        for j, s in enumerate(text_to_summarize[i:i+batch_size]):
            summary_partial[s] = summaries[j]
    with open(f"instagram_summaries_{part}.json", 'w') as summary_file:
        json.dump(summary_partial, summary_file)



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

    # with open("instagram_alpaca.json", 'w') as alpaca_file:
    #     json.dump(alpaca_instructions, alpaca_file, indent=2)