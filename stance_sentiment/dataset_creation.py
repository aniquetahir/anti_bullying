import pandas as pd
import os
from tqdm import tqdm
import json

SPLIT_TRAIN = 'train'
SPLIT_VAL = 'val'
SPLIT_TEST = 'test'

DATASET_TYPE_SENTIMENT = 'sentiment'
DATASET_TYPE_STANCE = 'stance'

STANCE_SAVE_PATH = "stance_alpaca.json"
SENTIMENT_SAVE_PATH = "sentiment_alpaca.json"

TWEETEVAL_DATASET_PATH = '/media/anique/Data/projects/tweeteval/datasets/'

INSTRUCTION_SENTIMENT_COT = """The input contains a tweet from social media.
First explain the tweet. Then, select the sentiment of the tweet given in the input. Select the sentiment from:
- negative
- neutral
- positive"""

INSTRUCTION_STANCE_COT = """The input contains a tweet from social media.
First explain the tweet. Then, select the stance of the tweet given in the input. Select the stance from:
- none
- against
- favor"""

MAPPING_STANCE = {
    0: 'none',
    1: 'against',
    2: 'favor'
}

MAPPING_SENTIMENT = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}


def get_dataset_generic(dataset_type, split, dataset_path=TWEETEVAL_DATASET_PATH):
    data_path = os.path.join(dataset_path, dataset_type, f'{split}_text.txt')
    labels_path = os.path.join(dataset_path, dataset_type, f'{split}_labels.txt')
    data = pd.read_csv(data_path, sep='\t', header=None)
    labels = pd.read_csv(labels_path, sep='\t', header=None)
    return data, labels


def get_sentiment_dataset(split, dataset_path=TWEETEVAL_DATASET_PATH):
    return get_dataset_generic(DATASET_TYPE_SENTIMENT, split, dataset_path)


def get_stance_dataset(split, dataset_path=TWEETEVAL_DATASET_PATH):
    stance_subcats = list(os.walk(os.path.join(dataset_path, DATASET_TYPE_STANCE)))[0][1]
    all_data = []
    all_labels = []

    for stance_subcat in stance_subcats:
        data, labels = get_dataset_generic(os.path.join(DATASET_TYPE_STANCE, stance_subcat), split, dataset_path)
        all_data.append(data)
        all_labels.append(labels)

    return pd.concat(all_data), pd.concat(all_labels)

# query the llama model
def create_alpaca_files():
    dataset_stance = {
        'train': [],
        'test': [],
        'val': []
    }

    dataset_sentiment = {
        'train': [],
        'test': [],
        'val': []
    }

    for split_name, split in tqdm([('train', SPLIT_TRAIN), ('test', SPLIT_TEST), ('val', SPLIT_VAL)]):
        x, y = get_stance_dataset(split)
        x, y = x.to_numpy(), y.to_numpy()
        iter_data = zip(x, y)
        for x_i, y_i in iter_data:
            dataset_stance[split].append(
                {
                    'instruction': INSTRUCTION_STANCE_COT,
                    'input': x_i.item(),
                    'output': MAPPING_STANCE[y_i.item()]
                }
            )

    for split_name, split in tqdm([('train', SPLIT_TRAIN), ('test', SPLIT_TEST), ('val', SPLIT_VAL)]):
        x, y = get_sentiment_dataset(split)
        x, y = x.to_numpy(), y.to_numpy()
        iter_data = zip(x, y)
        for x_i, y_i in iter_data:
            dataset_sentiment[split].append(
                {
                    'instruction': INSTRUCTION_SENTIMENT_COT,
                    'input': x_i.item(),
                    'output': MAPPING_SENTIMENT[y_i.item()]
                }
            )

    with open(STANCE_SAVE_PATH, 'w') as stance_save_file:
        json.dump(dataset_stance, stance_save_file, indent=2)

    with open(SENTIMENT_SAVE_PATH, 'w') as sentiment_save_file:
        json.dump(dataset_sentiment, sentiment_save_file, indent=2)


if __name__ == "__main__":
    sent_data, sent_labels = get_sentiment_dataset(SPLIT_TRAIN)
    stance_data, stance_labels = get_stance_dataset(SPLIT_TRAIN)
    print('data loaded')
    pass