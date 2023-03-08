from common import get_instagram_sessions
import json

if __name__ == "__main__":
    sessions = get_instagram_sessions()
    with open('sessions.json', 'w') as json_file:
        json.dump(sessions, json_file)