import pickle 
import os
from instagram.label_instagram import get_prompts_100_sessions
from common_lmql_phase1 import get_antibully_data
import numpy as np
import scipy
import pandas as pd


def get_statistics(labels, predictions):
    """
    labels: list of 0s and 1s
    predictions: list of 0s and 1s
    """
    # accuracy
    accuracy = sum([int(x == y) for x, y in zip(labels, predictions)]) / len(labels)
    # precision
    precision = sum([int(x == y and x == 1) for x, y in zip(labels, predictions)]) / sum(predictions)
    # recall
    recall = sum([int(x == y and x == 1) for x, y in zip(labels, predictions)]) / sum(labels)
    # f1
    f1 = 2 * precision * recall / (precision + recall)
    # auc
    # auc = scipy.stats.roc_auc_score(labels, predictions)
    return accuracy, precision, recall, f1

questions, instagram_prompts, _ = get_prompts_100_sessions()
def match_instagram_prompts(responses):
    
    anti_bully_label_data = get_antibully_data() # [{'session': ..., 'post_id': ...}]
    anti_bully_sessions = [x['session'] for x in anti_bully_label_data]

    anti_bully_label_data = [(i, x) for i, x in enumerate(questions) \
                             if {'session': x['session'], 'post_id': x['active_post_id'] - 1} in anti_bully_label_data]
    
    antibully_prompts = [instagram_prompts[i] for i, _ in anti_bully_label_data]
    bullying_prompts = [x for i, x in enumerate(instagram_prompts) if questions[i]['label'] == 1]
    np.random.seed(0)
    bullying_prompts = np.random.choice(bullying_prompts, len(antibully_prompts), replace=False).tolist()

    # add the relevant responses to the antibully prompts
    ab_mapped = []
    for p in antibully_prompts:
        for r in responses:
            if p == r[0].split('### Response:')[0] + '### Response:\n':
                ab_mapped.append((r[1], 0))
                break
    
    b_mapped = []
    for p in bullying_prompts:
        for r in responses:
            if p == r[0].split('### Response:')[0] + '### Response:\n':
                b_mapped.append((r[1], 1))
                break

    print(f'Found {len(ab_mapped)} antibully prompts and {len(b_mapped)} bullying prompts')

    labels = [1] * len(ab_mapped) \
        + [0] * len(ab_mapped) \
        + [1] * len(b_mapped) \
        + [0] * len(b_mapped)
    predictions = []
    predictions.extend([x[0]['antibullying'] == 'Yes' for x in ab_mapped])
    predictions.extend([x[0]['bullying'] == 'Yes' for x in ab_mapped])
    predictions.extend([x[0]['bullying'] == 'Yes' for x in b_mapped])
    predictions.extend([x[0]['antibullying'] == 'Yes' for x in b_mapped])

    accuracy, precision, recall, f1 = get_statistics(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def get_classification_statistics():
    with open('experiments/instagram_responses_part_llama2-7B_1_of_1.pkl', 'rb') as f:
        responses_7B = pickle.load(f)
    with open('experiments/instagram_responses_part_llama2-7B-merged_1_of_1.pkl', 'rb') as f:
        responses_7B_PEFT = pickle.load(f)
    with open('experiments/instagram_responses_part_llama2-13B_1_of_1.pkl', 'rb') as f:
        responses_13B = pickle.load(f)
    with open('experiments/instagram_responses_part_llama2-13B-phase2_1_of_1.pkl', 'rb') as f:
        responses_13B_PEFT = pickle.load(f)


    return {
        '7B': match_instagram_prompts(responses_7B),
        '7B-PEFT': match_instagram_prompts(responses_7B_PEFT),
        '13B': match_instagram_prompts(responses_13B),
        '13B-PEFT': match_instagram_prompts(responses_13B_PEFT)
    }


if __name__ == "__main__":
    statistics = get_classification_statistics()
    # create a csv from the statistics
    df = pd.DataFrame(statistics)
    df.to_csv('experiments/instagram_classification_results.csv')
