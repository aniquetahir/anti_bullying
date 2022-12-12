import pandas as pd
import numpy as np
from scipy import stats as st
import json
import os
import unicodedata
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from fa2 import ForceAtlas2
from functools import reduce


forceatlas2 = ForceAtlas2(
    # Behavior alternatives
    outboundAttractionDistribution=True,  # Dissuade hubs
    linLogMode=False,  # NOT IMPLEMENTED
    adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
    edgeWeightInfluence=1.0,

    # Performance
    jitterTolerance=1.0,  # Tolerance
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    multiThreaded=False,  # NOT IMPLEMENTED

    # Tuning
    scalingRatio=2.0,
    strongGravityMode=False,
    gravity=1.0,

    # Log
    verbose=True)


instagram_data_dir = "instagram_100_sessions/Compiled and Totaled Sessions 2020"

def get_session_info(excel_file: pd.ExcelFile, sheet_name: str):
    str_label = sheet_name.split('-')[1].strip()
    int_label = 0 if str_label.lower() == 'no' else 1

    sheet_content = pd.read_excel(excel_file, sheet_name)
    columns = sheet_content.columns
    poster_name = sheet_content[sheet_content.columns[0]][0]
    poster_name = poster_name.strip()
    poster_comment = sheet_content[sheet_content.columns[2]][0]

    # parse all the comments
    comment_posters= sheet_content[2:][columns[1]].tolist()
    comment_texts  = sheet_content[2:][columns[2]].tolist()
    comment_labels = sheet_content[2:][columns[4]].tolist()
    directions   = [list(sheet_content[2:][columns[x]].to_list()) for x in [35, 36, 37]]
    directions = list(zip(*directions))
    comments = []
    # create a structure from the extracted data
    for i in range(len(comment_posters)):
        # print(directions[i])
        try:
            comments.append({
                'user': comment_posters[i].strip(),
                'to': [x.strip().replace('@', '') if not pd.isna(x) and type(x) == str else '' for x in directions[i]],
                'content': comment_texts[i].strip(),
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
        'label': int_label,
        'comments': comments
    }

    return session

if __name__ == "__main__":
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

    # get the set of unique users
    users = []

    for session in sessions:
        users.append(session['user'])
        users.extend([x['user'] for x in session['comments']])
        # Also add the people mentioned
        users.extend([y for x in session['comments'] for y in x['to']])

    users = list(set(users))

    user_bully_dict = defaultdict(lambda: {0: 0, 1: 0})
    # For each user judge whether they were involved in bullying or not
    for session in sessions:
        for comment in session['comments']:
            try:
                user = comment['user']
                label = comment['label']
                user_bully_dict[user][label] += 1
            except Exception as e:
                print(comment)

    user_bully_perc = {}
    for k, v in user_bully_dict.items():
        if v[0] + v[1] == 0:
            perc = 0
        else:
            perc = v[1]/(v[0] + v[1])

        user_bully_perc[k] = perc

    bullies = [k for k, v in user_bully_perc.items() if v>0]
    nonbullies = [k for k, v in user_bully_perc.items() if v == 0]


    user_to_id = {}
    id_to_user = {}
    for i, user in enumerate(users):
        user_to_id[user] = i
        id_to_user[i] = user


    edges = []
    for session in sessions:
        for comment in session['comments']:
            try:
                to_mapped = [user_to_id[x] for x in comment['to'] if x != '']
                if len(to_mapped) == 0:
                    continue
                e_to = st.mode(to_mapped).mode.item()
                e_from = user_to_id[comment['user']]
                edges.append((e_from, e_to))
            except Exception as e:
                print(e)
                print(to_mapped)

    G = nx.DiGraph()
    G.add_nodes_from(list(id_to_user.keys()))
    G.add_edges_from(edges)

    lcc = reduce(lambda a, b: b if len(b) > len(a) else a, nx.weakly_connected_components(G)) # .connected_components(G))
    G = G.subgraph(lcc)

    pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    bullies_idx = [user_to_id[x] for x in bullies]
    nonbullies_idx = [user_to_id[x] for x in nonbullies]
    bullies_idx = [x for x in bullies_idx if x in G.nodes]
    nonbullies_idx = [x for x in nonbullies_idx if x in G.nodes]

    plt.clf()
    nx.draw_networkx_nodes(G, pos, nodelist=bullies_idx, node_color='#ff0000', node_size=3)
    nx.draw_networkx_nodes(G, pos, nodelist=nonbullies_idx, node_color='#0000ff', node_size=3)
    nx.draw_networkx_edges(G, pos)
    plt.show()
    plt.savefig('fig.svg')


    pass