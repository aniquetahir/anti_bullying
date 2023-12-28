from common import merge_json_list_files
import os

if __name__ == "__main__":
    # get the summary file names
    summary_files = [f for f in os.listdir('.') if f.startswith('instagram_summaries_part')]
    merge_json_list_files(summary_files, 'instagram_summaries.json')