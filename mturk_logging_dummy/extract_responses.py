import json
# import for extracting url query string
from urllib.parse import urlparse, parse_qs

def extract_responses(filepath: str):
    log_contents = []
    with open(filepath, 'r') as log_file:
        log_contents = log_file.readlines()

    data = []

    for line in log_contents:
        if line.startswith('mturk_answers='):
            # extract variables from url query string
            url_query_string = parse_qs(line)
            # extract the mturk_answers variable
            mturk_answers = url_query_string['mturk_answers'][0]
            # convert to json
            mturk_answers_json = json.loads(mturk_answers)
            session_id = int(mturk_answers_json['sessionId'])
            for ab in mturk_answers_json['antiBullyIdxs']:
                data.append({
                    'session': session_id,
                    'post_id': int(ab)
                })
    return data

if __name__ == "__main__":
    responses = extract_responses('request.log')
    # save the responses
    with open('mturk_logging_dummy/responses.json', 'w') as responses_file:
        json.dump(responses, responses_file, indent=2)