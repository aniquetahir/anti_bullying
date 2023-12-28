import json
from instagram_utils.common import create_session_questions
from common_generative import generate_prompt
from os.path import join as pjoin
from augmentation import get_session_tabels_and_labels
from common import get_instagram_sessions
from instagram_utils.common import instruction

INSTAGRAM_DATA_FILE = "instagram/instagram_sessions_full.json"
BATCH_SIZE = 15
WEIGHTS_FOLDER = "/media/anique/Data/projects/llama-weights/"
model_path = pjoin(WEIGHTS_FOLDER, 'llama2-7B-merged')
tokenizer_path = pjoin(WEIGHTS_FOLDER, 'llama2-7B')

def get_prompts(sessions, window_size, return_sessions=False):
    prompts = []
    prompt_sessions = []
    for session in sessions:
        questions = create_session_questions(session, window_size)
        for question in questions:
            prompts.append(generate_prompt(question['instruction'], '```\n' + question['input'] + '\n```'))
            prompt_sessions.append(session['session'])
    return (prompts, prompt_sessions) if return_sessions else prompts

def get_all_prompts(return_sessions=False):
    with open(INSTAGRAM_DATA_FILE, 'r') as f:
        sessions = json.load(f)
    prompts = get_prompts(sessions, 20, return_sessions=return_sessions)
    return prompts


def get_prompts_100_sessions(window_size=10):
    sessions = get_instagram_sessions()
    session_dataframes = [(s_i, get_session_tabels_and_labels(x, window_size=window_size)) for s_i, x in enumerate(sessions)]
    inputs = []
    labels = []
    sessions = []
    active_post_ids = []
    session_dataframes = [(s_i, y) for s_i, x in session_dataframes for y in x]
    for s_i, df in session_dataframes:
        # get the last label for field for the dataframe
        label = df['label'].iloc[-1]
        active_post_id = df['post_id'].iloc[-1]

        # get all the columns except label
        df.drop(columns=['label'], inplace=True)

        # convert the dataframe to a csv
        input_text = df.to_csv(index=False)
        input_text = ((input_text.replace('"""', "")
                      .replace('[', '')
                      .replace(']', ''))
                      .replace('{', '').replace('}', ''))
        inputs.append(input_text)
        labels.append(label)
        sessions.append(s_i)
        active_post_ids.append(active_post_id)

    questions = []
    for i, input_csv in enumerate(inputs):
        questions.append({
            'session': sessions[i],
            'input': input_csv,
            'instruction': instruction,
            'label': labels[i],
            'active_post_id': active_post_ids[i]
        })

    prompts = []
    prompt_labels = []
    for question in questions:
        prompts.append(generate_prompt(question['instruction'], '```\n' + question['input'] + '\n```'))
        prompt_labels.append(question['label'])
    return questions, prompts, prompt_labels

def generate_raw_responses(sessions, window_size):
    # prompts = get_prompts_100_sessions()
    prompts = get_prompts(sessions, window_size)

    return prompts

if __name__ == "__main__":
    get_prompts_100_sessions()