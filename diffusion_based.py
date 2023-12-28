import pandas as pd
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from common import get_instagram_sessions, get_text_embedding, get_text_embeddings_splitted
from collections import  defaultdict
from scipy.spatial.distance import cosine, euclidean
import plotly.express as px
model_id = "stabilityai/stable-diffusion-2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_word_lists():
    with open('apologetic.txt', 'r') as apologetic_file:
        apologetic_context = apologetic_file.readlines()
    with open('comforting.txt', 'r') as comfort_file:
        comfort_context = comfort_file.readlines()
    with open('silencing.txt', 'r') as stopping_file:
        stopping_context = stopping_file.readlines()

    return apologetic_context, comfort_context, stopping_context

if __name__ == "__main__":
    sessions = get_instagram_sessions()
    all_comments = [y['content'] for x in sessions for y in x['comments']]
    # Use the Euler scheduler here instead
    # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16, max_len=300, truncation=True, padding=True)
    # pipe.tokenizer.model_max_length = 300
    # pipe = pipe.to("cuda")

    apol, comf, stop = get_word_lists()
    apol, comf, stop = [[x.strip() for x in y] for y in [apol, comf, stop]]

    # pipe._encode_prompt(apol, device, 1, True, None)
    # Get the encodings for all the words
    architype_encodings = []
    for l in [apol, comf, stop]:
        m_l = get_text_embedding(l)
        architype_encodings.append(m_l)

    # Encode the instagram session texts
    comment_encodings = get_text_embeddings_splitted(all_comments, 500)

    # Compare the encoded comments with the architypes
    comment_stats = defaultdict(list)
    for i_c, ce in enumerate(comment_encodings):
        for i, arch in enumerate(architype_encodings):
            min_cos = min_euc = float('inf')
            mean_cos = mean_euc = 0
            for arch_i in arch:
                cos_dist = cosine(arch_i, ce)
                euc_dist = euclidean(arch_i, ce)
                min_cos = cos_dist if cos_dist<min_cos else min_cos
                min_euc = euc_dist if euc_dist<min_euc else min_euc
                mean_cos += cos_dist
                mean_euc += euc_dist
            mean_euc, mean_cos = mean_euc/len(arch), mean_cos/len(arch)
            comment_stats[i_c].append({
                'arch': i,
                'min_cos': min_cos,
                'min_euc': min_euc,
                'mean_cos': mean_cos,
                'mean_euc': mean_euc
            })

    comment_data = []
    for i, c in enumerate(all_comments):
        comment_datum = {}
        for v in comment_stats[i]:
            arch = v['arch']
            for k1, v1 in v.items():
                if k1 == 'arch':
                    continue
                comment_datum[f'{k1}_{arch}'] = v1
        comment_datum['comment'] = c
        comment_data.append(comment_datum)

    df_comments = pd.DataFrame(comment_data)

    # Visualize using plotly
    print('test')


    # Find the average similarity between the architypes (cosine distance and euclidean distance)

    prompt = "a photo of an astronaut riding a horse on mars"

    text_embedding = pipe._encode_prompt(prompt, device, 1, True, None)[1]


    print('Hello World')