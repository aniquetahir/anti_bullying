import os
import json
import numpy as np

ALPACA_PROJECT_LOCATION = '/home/anique/projects/alpaca-lora'
def create_merged_dataset(alpaca_dataset_location, instagram_dataset_location, fourchan_dataset_location, output_location):
    # create the merged dataset
    final_dataset = []
    # load the alpaca dataset
    with open(alpaca_dataset_location, 'r') as json_file:
        alpaca_dataset = json.load(json_file)
    # load the instagram dataset
    with open(instagram_dataset_location, 'r') as json_file:
        instagram_dataset = json.load(json_file)
    # load the fourchan dataset
    with open(fourchan_dataset_location, 'r') as json_file:
        fourchan_dataset = json.load(json_file)
    # merge the datasets

    len_alpaca, len_insta, len_4chan = len(alpaca_dataset), len(instagram_dataset), len(fourchan_dataset)
    # find the max length for each dataset such that the merge satisfies 20/40/40 percent split
    l_a = min(len_alpaca, len_4chan)
    l_i = int(l_a * 8)
    if l_i > len_insta:
        l_i = len_insta
        l_a = len_insta // 8

    # sample the datasets
    alpaca_idxs = np.arange(len_alpaca)
    alpaca_choices = np.random.choice(alpaca_idxs, l_a, replace=False)
    fourchan_choices = np.random.choice(np.arange(len_4chan), l_a, replace=False)

    instagram_choices = np.random.choice(np.arange(len_insta), l_i, replace=False)

    for i in alpaca_choices:
        final_dataset.append(alpaca_dataset[i])

    for i in fourchan_choices:
        final_dataset.append(fourchan_dataset[i])

    for i in instagram_choices:
        final_dataset.append(instagram_dataset[i])

    # shuffle the dataset
    data_idxs = np.arange(len(final_dataset))
    np.random.shuffle(data_idxs)
    final_dataset_shuffled = []
    for i in data_idxs:
        final_dataset_shuffled.append(final_dataset[i])

    # save the dataset
    with open(output_location, 'w') as json_file:
        json.dump(final_dataset_shuffled, json_file, indent=2)


if __name__ == "__main__":
    alp_dataset_loc = os.path.join(ALPACA_PROJECT_LOCATION, 'alpaca_data_cleaned.json')
    insta_dataset_loc = 'instagram_alpaca.json'
    fourchan_dataset_loc = os.path.join(ALPACA_PROJECT_LOCATION, 'bard_data_alpaca_augmented.json')

    output_loc = 'merged_dataset_insta_4chan.json'
    create_merged_dataset(alp_dataset_loc, insta_dataset_loc, fourchan_dataset_loc, output_loc)
