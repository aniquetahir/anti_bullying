import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
from tqdm import tqdm
import json
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = '/media/anique/Data/datasets/instagram/all_matched_labeled_data.csv'

def convert_session():
    pass


# Comment format <font ...>username</font> text <font ...>username</font>
def clean_comment(comment_text: str):
    if 'created at:' not in comment_text:
        return None

    soup = BeautifulSoup(comment_text, 'html.parser')
    font_tags = []
    for font in soup.find_all('font'):
        font_tags.append(font)

    # the first font tag is the author of the comment
    author = font_tags[0].text.strip()

    # For the rest of the font tags, we need to replace them with the text content
    for font in font_tags[1:]:
        font.replace_with(font.text)

    # Remove the first font tag
    font_tags[0].decompose()

    comment_text = soup.text.strip()

    # date format (created at:2012-06-03 12:44:40)
    # extract the date as time from epoch from the comment text
    date_regex = r'created at:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    date_match = re.search(date_regex, comment_text)
    if date_match:
        date_span = date_match.span()
        date = comment_text[date_span[0]:date_span[1]]
        comment_text = comment_text[:date_span[0] - 1] + comment_text[date_span[1] + 1:]
    else:
        date = ''

    # convert the date to time from epoch
    date = datetime.strptime(date, 'created at:%Y-%m-%d %H:%M:%S')
    date = date.timestamp()

    # remove the 'created at:' from the comment text
    comment_text = comment_text.replace('created at:', '').strip()

    if 'Marley' in comment_text:
        print('tag found')

    return {
        'author': author,
        'text': comment_text,
        'date': date
    }


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    sessions = []
    records = df.to_dict('records')
    comment_nums = []
    for i, record in enumerate(tqdm(records)):
        # get the comments
        comments = [v for k, v in record.items() if re.match('^[0-9]+$', k)]
        # clean the comments
        comments = [clean_comment(comment) for comment in comments if type(comment) == str]
        # remove the None values
        comments = [comment for comment in comments if comment is not None]
        # sort the comments by date
        comments = sorted(comments, key=lambda x: x['date'])
        # add the comments to the sessions
        session = {
            'session': str(i),
            'user': BeautifulSoup(record['owner_id'], 'html.parser').text.strip(),
            'content': BeautifulSoup(str(record['owner_cmnt']), 'html.parser').text.strip(),
            'date': record['_created_at'],
            'comments': []
        }

        if len(comments) == 0:
            continue
        comment_nums.append(len(comments))
        for comment in comments:
            session['comments'].append({
                'user': comment['author'],
                'content': comment['text'],
                'date': datetime.fromtimestamp(comment['date']).strftime('%Y-%m-%d %H:%M:%S')
            })
        sessions.append(session)

    fig = sns.histplot(comment_nums)
    plt.xlabel('Number of comments')
    plt.ylabel('Number of sessions')

    plt.savefig('instagram_dataset_comment_stats.pdf')
    plt.show()

    print(f'Number of sessions: {len(sessions)}')

    # save sessions
    with open('instagram_sessions_full.json', 'w') as f:
        json.dump(sessions, f, indent=2)
