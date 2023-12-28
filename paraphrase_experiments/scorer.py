import os.path

import nltk
from rouge_score import rouge_scorer
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

from transformers import AutoTokenizer, AutoModel
import torch
import enchant
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import pickle
from tqdm import tqdm
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np


SAVE_PATH = 'paraphrase_experiments/compiled_statistics.json'
tokenizer, model = None, None
def calculate_semantic_similarity(str1, str2, model_name='bert-base-uncased'):
    global tokenizer, model
    if tokenizer is None:
        # Load pre-trained model tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode text
    inputs_1 = tokenizer(str1, return_tensors='pt', padding=True, truncation=True)
    inputs_2 = tokenizer(str2, return_tensors='pt', padding=True, truncation=True)

    if model is None:
        # Load pre-trained model
        model = AutoModel.from_pretrained(model_name)

    # Forward pass, get hidden states
    with torch.no_grad():
        outputs_1 = model(**inputs_1)
        outputs_2 = model(**inputs_2)

    # Only take the output embeddings from the last layer
    embeddings_1 = outputs_1.last_hidden_state.mean(dim=1)
    embeddings_2 = outputs_2.last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(
        embeddings_1.cpu().numpy(),
        embeddings_2.cpu().numpy()
    )

    return cosine_sim[0][0]


def calculate_bleu(reference, candidate):
    smoothing = SmoothingFunction()
    reference = [word_tokenize(reference)]
    candidate = word_tokenize(candidate)
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing.method1)  # You can change the method as per your requirement
    return bleu_score


def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def calculate_jaccard_similarity(str1, str2):
    tokens1 = set(word_tokenize(str1))
    tokens2 = set(word_tokenize(str2))
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1) + len(tokens2) - intersection
    return intersection / union


def calculate_cosine_distance(str1, str2):
    vectorizer = CountVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cosine_distance = cosine_distances(vectors)
    return cosine_distance[0, 1]  # The value at [0,1] or [1,0] will give the cosine distance between the two vectors

def calculate_statistics(sentence_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    stats = {
        'bleu': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'jaccard': [],
        'cosine': [],
        'semantic': [],
        'levenstein': [],
        'levenstein_ratio': []
    }
    for sp in tqdm(sentence_pairs):
        stats['bleu'].append(calculate_bleu(sp[0], sp[1]))
        rouge_scores = calculate_rouge(sp[0], sp[1])
        stats['rouge1'].append(rouge_scores['rouge1'].fmeasure)
        stats['rouge2'].append(rouge_scores['rouge2'].fmeasure)
        stats['rougeL'].append(rouge_scores['rougeL'].fmeasure)
        stats['jaccard'].append(calculate_jaccard_similarity(sp[0], sp[1]))
        stats['cosine'].append(calculate_cosine_distance(sp[0], sp[1]))
        stats['semantic'].append(calculate_semantic_similarity(sp[0], sp[1]))
        stats['levenstein'].append(nltk.edit_distance(sp[0], sp[1]))
        stats['levenstein_ratio'].append(nltk.edit_distance(sp[0], sp[1]) / max(len(sp[0]), len(sp[1])))
    return {k: [float(x) for x in v] for k, v in stats.items()}


def calculate_stats(samples_pkl: str):
    with open(samples_pkl, 'rb') as f:
        samples = pickle.load(f)

    stats = calculate_statistics(samples)
    return stats


if __name__ == "__main__":
    model_stats = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, 'r') as f:
            model_stats = json.load(f)
    else:
        # open the 7b non exemplars file
        model_stats['7b_exemplars'] = calculate_stats('paraphrases_7b.pkl')
        model_stats['7b_non_exemplars'] = calculate_stats('paraphrases_7b_noexemplars.pkl')
        model_stats['13b_exemplars'] = calculate_stats('paraphrases_13b.pkl')
        model_stats['13b_non_exemplars'] = calculate_stats('paraphrases_13b_noexemplars.pkl')
        model_stats['chatgpt_exemplars'] = calculate_stats('paraphrases_chatgpt_exemplars.pkl')
        model_stats['chatgpt_non_exemplars'] = calculate_stats('paraphrases_chatgpt_noexemplars.pkl')
        model_stats['gpt2_exemplars'] = calculate_stats('paraphrases_gpt2_exemplars.pkl')
        model_stats['gpt2_non_exemplars'] = calculate_stats('paraphrases_gpt2_noexemplars.pkl')

        # save all the statistics in a json file
        with open(SAVE_PATH, 'w') as f:
            json.dump(model_stats, f, indent=4)


    df_dict = {'model': [], 'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [],
               'jaccard': [], 'cosine': [], 'semantic': [], 'levenstein': [], 'levenstein_ratio': [], 'type': []}

    # create a dataframe with the statistics
    for model, stats in model_stats.items():
        for i, (metric, values) in enumerate(stats.items()):
            for value in values:
                if i ==0:
                    df_dict['model'].append(model.split('_')[0].upper())
                df_dict[metric].append(value)
                if i == 0:
                    if 'non_exemplars' in model:
                        df_dict['type'].append('non_exemplar')
                    else:
                        df_dict['type'].append('exemplar')



    # for model, stats in model_stats.items():
    #     for metric, values in stats.items():
    #         for value in values:
    #             df_dict['model'].append(model.split('_')[0].upper())
    #             df_dict['metric'].append(metric)
    #             df_dict['value'].append(value)
    #             if 'non_exemplars' in model:
    #                 df_dict['type'].append('non_exemplar')
    #             else:
    #                 df_dict['type'].append('exemplar')

    # create the dataframe
    df = pd.DataFrame(df_dict)
    # draw the violin plot for each metric in a grid of 3 rows and 3 columns
    # the x axis will be the model name, and the y axis will be the metric value
    # the hue will be the type of the sample (exemplar or non exemplar)
    # g = sns.FacetGrid(df, col='metric', row='type', height=4, aspect=1.5)
    # g.map(sns.violinplot, 'model', 'value', hue=type, split=True, inner='quart')
    # plt.show()

    # for each model, calculate the mean and std of each metric
    for model, stats in model_stats.items():
        print(model)
        for metric, values in stats.items():
            print(metric, np.mean(values), np.std(values))

    # create a csv table with the aggregated statistics
    df_agg = df.groupby(['model', 'type']).agg({'bleu': ['mean', 'std'],
                                                'rouge1': ['mean', 'std'],
                                                'rouge2': ['mean', 'std'],
                                                'rougeL': ['mean', 'std'],
                                                'jaccard': ['mean', 'std'],
                                                'cosine': ['mean', 'std'],
                                                'semantic': ['mean', 'std'],
                                                'levenstein': ['mean', 'std'],
                                                'levenstein_ratio': ['mean', 'std']})
    df_agg.to_csv('stats.csv')

    fig = px.scatter(df, x="levenstein_ratio", y="semantic", color="model",
                     labels={'levenstein_ratio': 'Levenshtein ratio', 'semantic': 'Semantic similarity'},
                 marginal_x="violin", marginal_y="violin", symbol="type")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    # fig.update_xaxes(title_text='Levenstein ratio')
    # fig.update_yaxes(title_text='Semantic similarity')
    # fig.update_xaxes(autorange='reversed')
    # fig.update_yaxes(autorange='reversed')
    fig.write_image('paraphrase_experiments/experiment1.pdf')
    fig.show()

    # draw the violin plot
    # filter data for the bleu metric
    df_bleu = df[df['metric'] == 'bleu']
    plt_bleu = sns.violinplot(x='model', y='value', hue='type', data=df_bleu, split=True, inner='quart')
    # set the y axis title to be the metric name
    plt.ylabel('BLEU')
    plt.show()

    # filter data for the rouge1 metric
    df_rouge1 = df[df['metric'] == 'rouge1']
    plt_rouge1 = sns.violinplot(x='model', y='value', hue='type', data=df_rouge1, split=True, inner='quart')
    plt.ylabel('ROUGE-1')
    plt.show()

    # filter data for the rouge2 metric
    df_rouge2 = df[df['metric'] == 'rouge2']
    plt_rouge2 = sns.violinplot(x='model', y='value', hue='type', data=df_rouge2, split=True, inner='quart')
    plt.ylabel('ROUGE-2')
    plt.show()

    # filter data for the rougeL metric
    df_rougeL = df[df['metric'] == 'rougeL']
    plt_rouge_l = sns.violinplot(x='model', y='value', hue='type', data=df_rougeL, split=True, inner='quart')
    plt.ylabel('ROUGE-L')
    plt.show()

    # filter data for the jaccard metric
    df_jaccard = df[df['metric'] == 'jaccard']
    plt_jaccard = sns.violinplot(x='model', y='value', hue='type', data=df_jaccard, split=True, inner='quart')
    plt.ylabel('Jaccard')
    plt.show()

    # filter data for the cosine metric
    df_cosine = df[df['metric'] == 'cosine']
    plt_cosine = sns.violinplot(x='model', y='value', hue='type', data=df_cosine, split=True, inner='quart')
    plt.ylabel('Cosine')
    plt.show()

    # filter data for the semantic metric
    df_semantic = df[df['metric'] == 'semantic']
    plt_semantic = sns.violinplot(x='model', y='value', hue='type', data=df_semantic, split=True, inner='quart')
    plt.ylabel('Semantic')
    plt.show()

    # filter data for the levenstein metric
    df_levenstein = df[df['metric'] == 'levenstein']
    plt_levenstein = sns.violinplot(x='model', y='value', hue='type', data=df_levenstein, split=True, inner='quart')
    plt.ylabel('Levenstein')
    plt.show()

    # filter data for the levinshtein ratio metric
    df_levinshtein_ratio = df[df['metric'] == 'levenstein_ratio']
    plt_levenstein_ratio = sns.violinplot(x='model', y='value', hue='type', data=df_levinshtein_ratio, split=True, inner='quart')
    plt.ylabel('Levinshtein Ratio')
    plt.show()


    reference = "I love machine learning."
    candidate = "I adore machine learning."



    # Calculate BLEU Score
    bleu = calculate_bleu(reference, candidate)
    print(f"BLEU Score: {bleu}")

    # Calculate ROUGE Score
    rouge = calculate_rouge(reference, candidate)
    print(f"ROUGE Score: {rouge}")

    # Calculate Jaccard Similarity
    jaccard_similarity = calculate_jaccard_similarity(reference, candidate)
    print(f"Jaccard Similarity: {jaccard_similarity}")

    # Calculate Cosine Distance
    cosine_distance = calculate_cosine_distance(reference, candidate)
    print(f"Cosine Distance: {cosine_distance}")

    # Test the function
    similarity = calculate_semantic_similarity(reference, candidate)

    print(f"Semantic Similarity: {similarity}")

    levenstein = nltk.edit_distance(reference, candidate)
    print(f"Levenstein Distance: {levenstein}")