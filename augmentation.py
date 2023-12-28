from common import create_session_table
import numpy as np
import pandas as pd

def get_augmentation_questions(aug_fn, session, window_size: int):
    start_idxs = np.arange(0, len(session['comments']), window_size)
    session_answers = []
    for i in start_idxs:
        _, df = create_session_table(session, till=i+window_size, window_size=window_size)
        session_answers.append(aug_fn(df))

    return session_answers

def get_session_tabels_and_labels(session, window_size: int):
    start_idxs = np.arange(1, len(session['comments']))
    session_answers = []
    for i in start_idxs:
        _, df = create_session_table(session, till=i, window_size=window_size, return_labels=True)
        session_answers.append(df)
    return session_answers

# What is the name of the author for the post with id {post_id}?
def create_name_of_author_dataset(session_df: pd.DataFrame):
    authors = list(set(session_df['author'].tolist()))
    answers = [] # post, author
    for author in authors:
        # get the posts for the author
        author_posts = session_df[session_df['author'] == author]['post_id'].tolist()
        # select a random post
        for pid in author_posts:
            answers.append({
                'post': pid,
                'author': author
            })
    return session_df.to_csv(index=False), answers

# Give a summary of the posts by the author {author}.
def summarize_posts_by_author(session_df: pd.DataFrame, summary_fn=None):
    authors = list(set(session_df['author'].tolist()))
    answers = [] # author, summary
    for author in authors:
        author_posts = session_df[session_df['author'] == author]['comment'].tolist()
        summary = summary_fn('\n'.join(author_posts))
        answers.append({
            'author': author,
            'summary': summary
        })
    return session_df.to_csv(index=False), answers

# Who is the author who has the following views:\n {summary}?
def who_said_that(session_df, summary_fn=None):
    return summarize_posts_by_author(session_df, summary_fn=summary_fn)

# What is the ID of the post which said the following:\n {paraphrase}?
def post_which_said(session_df, paraphrase_fn=None):
    post_ids = session_df['post_id'].tolist()
    answers = [] # para, postid
    for pid in post_ids:
        post_content = session_df[session_df['post_id'] == pid]['comment'].tolist()[0]
        paraphrase = paraphrase_fn(post_content)
        answers.append({
            'paraphrase': paraphrase,
            'post_id': pid
        })
    return session_df.to_csv(index=False), answers