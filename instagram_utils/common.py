import pandas as pd

def create_session_table(session, till=-1, window_size=10):
    if till == 0:
        raise Exception('till cannot be 0')

    comments = []
    users = []

    users.append(session['user'])
    comments.append(session['content'])

    for c in session['comments']:
        users.append(c['user'])
        comments.append(c['content'])

    if till != -1:
        start_idx = max(0, till - window_size)
        end_idx = min(len(comments) -1 , till + 1)
    else:
        start_idx = 0
        end_idx = len(comments) - 1

    table_content = [{
        'post_id': 1,
        'author': users[0],
        'comment': comments[0],
        'active': 0
    }]

    # print(f'start: {start_idx}, end: {end_idx}')
    # table will contain post_id, author, comment, is_active
    for i in range(start_idx + 1, end_idx):
        if till!=-1 and i>till:
            break
        table_content.append({
            'post_id': i + 1,
            'author': users[i],
            'comment': comments[i],
            'active': 0 if i!=till else 1
        })
    dataframe = pd.DataFrame(table_content)
    active = dataframe['active'].to_numpy()
    active[-1] = 1
    dataframe['active'] = active
    context_csv = dataframe.to_csv(index=False)
    return context_csv, dataframe


instruction = """You have been provided with a CSV file containing a social media conversation.
For this task, you should only make assumptions about posters based on the provided CSV input. The posts in the input are
in the order of date posted i.e. replies do not occur before posts being replied to.
Answer the following questions:
a) Summarize the discourse revolving around the active post ID.
b) Is the active post considered bullying?
c) Is the active post considered anti-bullying?
d) If it is bullying, explain why?
e) If it is anti-bullying, explain why?
f) If it is neither, explain why?
"""

def session_obj_to_csvs(session, window_size=10):
    # create a table for each sliding window
    num_comments = len(session['comments'])
    csvs = []
    for i in range(0, num_comments, window_size):
        input_csv, _ = create_session_table(session, till=i + window_size, window_size=window_size)
        csvs.append(input_csv)
    return csvs

def create_session_questions(session, window_size=10):
    questions = []
    inputs = session_obj_to_csvs(session, window_size=window_size)
    for input_csv in inputs:
        questions.append({
            'input': input_csv,
            'instruction': instruction
        })
    return questions

