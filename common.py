import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import re

from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification, AutoModel

instagram_data_dir = "instagram_100_sessions/Compiled and Totaled Sessions 2020"
time_regex = "[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+"

def get_session_info(excel_file: pd.ExcelFile, sheet_name: str):
    str_label = sheet_name.split('-')[1].strip()
    int_label = 0 if str_label.lower() == 'no' else 1

    sheet_content = pd.read_excel(excel_file, sheet_name)
    columns = sheet_content.columns
    poster_name = sheet_content[sheet_content.columns[0]][0]
    poster_name = poster_name.strip()
    poster_comment = sheet_content[sheet_content.columns[2]][0]
    poster_time = sheet_content[sheet_content.columns[3]][0]
    poster_time = poster_time if type(poster_time) == str else ''
    t = re.search(time_regex, poster_time)
    if t:
        t_span = t.span()
        poster_time = poster_time[t_span[0]:t_span[1]]

    # parse all the comments
    comment_posters= sheet_content[2:][columns[1]].tolist()
    comment_texts  = sheet_content[2:][columns[2]].tolist()
    comment_time = sheet_content[2:][columns[3]].tolist()
    comment_labels = sheet_content[2:][columns[4]].tolist()
    directions   = [list(sheet_content[2:][columns[x]].to_list()) for x in [35, 36, 37]]
    directions = list(zip(*directions))
    comments = []
    # create a structure from the extracted data
    for i in range(len(comment_posters)):
        # print(directions[i])
        try:
            c_time = comment_time[i]
            c_t_match = re.search(time_regex ,c_time)
            if c_t_match:
                c_t_span = c_t_match.span()
                c_time = c_time[c_t_span[0]:c_t_span[1]]
            comments.append({
                'user': comment_posters[i].strip(),
                'to': [x.strip().replace('@', '') if not pd.isna(x) and type(x) == str else '' for x in directions[i]],
                'content': comment_texts[i].strip(),
                'time': c_time,
                'label': int(comment_labels[i])
            })
        except Exception as e:
            print('=' * 10)
            print(e)
            print('=' * 10)
            print(comment_posters[i])
            print(directions[i])
            print(comment_texts[i])
            print(comment_labels[i])

    session = {
        'session': sheet_name,
        'user': poster_name,
        'content': poster_comment.strip(),
        'time': poster_time.strip(),
        'label': int_label,
        'comments': comments
    }

    return session

def get_instagram_sessions():
    xl_file_names = [os.path.join(instagram_data_dir, x) for x in os.listdir(instagram_data_dir) if x.endswith('.xlsx') and 'Totals' not in x]
    xl_files = [os.path.join(instagram_data_dir, x) for x in os.listdir(instagram_data_dir) if x.endswith('.xlsx') and 'Totals' not in x]
    xl_files = [pd.ExcelFile(x) for x in xl_files]

    sessions = []
    for file_i, xl_file in enumerate(xl_files):
        print(f'Doing {xl_file_names[file_i]}')
        for sheet in xl_file.sheet_names:
            print(f'Doing sheet {sheet}')
            session_info = get_session_info(xl_file, sheet)
            sessions.append(session_info)

    return sessions

def get_embeddings(tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification, corpus, submodule_name='bert'):
    with torch.no_grad():
        try:
            inputs = tokenizer(corpus, padding=True)['input_ids']
            submodule = model.get_submodule(submodule_name)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            hidden_states = submodule(torch.tensor(torch.tensor(inputs).to(device)))[0][:, -4:, :]
            cat_state = hidden_states.reshape(hidden_states.shape[0], -1)
            embeddings = cat_state
        except Exception as e:
            print(corpus)
            print(e)
    return embeddings.to('cpu')

def get_text_embedding(text_list):
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    roberta_model = AutoModelForSequenceClassification.from_pretrained('roberta-large')
    if torch.cuda.is_available():
        roberta_model = roberta_model.to('cuda')
    batch_embeddings = get_embeddings(roberta_tokenizer, roberta_model, text_list, submodule_name='roberta')
    return batch_embeddings

def get_text_embeddings_splitted(text_list, batch_size):
    embeddings_list = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        embeddings_list.append(get_text_embedding(batch))

    return torch.vstack(embeddings_list)

def get_simcse_embeddings(text_list):
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        emb = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return emb.to('cpu')
def get_simcse_text_embeddings_splitted(text_list, batch_size):
    embeddings_list = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]
        embeddings_list.append(get_simcse_embeddings(batch))

    return torch.vstack(embeddings_list)
