import requests
import base64
import os
import logging
from dotenv import load_dotenv
import time

load_dotenv()

ACCESS_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = os.getenv('REPO_OWNER')
REPO_NAME = os.getenv('REPO_NAME')
BRANCH = os.getenv('BRANCH')
BASE_URL = os.getenv('BASE_URL')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

if not all([ACCESS_TOKEN, REPO_OWNER, REPO_NAME, BRANCH, BASE_URL, OUTPUT_DIR]):
    raise ValueError("Missing one or more required environment variables.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

file_counter = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_store(path='', depth=0, max_depth=10):
    global file_counter
    if depth > max_depth:
        return

    url = f'{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}'
    headers = {'Authorization': f'token {ACCESS_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers, params={'ref': BRANCH})

    logging.info(f"Request URL: {url}")
    logging.info(f"Response status code: {response.status_code}")

    if response.status_code == 200:
        contents = response.json()
        if isinstance(contents, list):
            for item in contents:
                if item['type'] == 'file' and item['name'].lower().endswith('.md'):
                    save_file(item, headers)
                elif item['type'] == 'dir':
                    logging.info(f"Entering directory: {item['path']}")
                    fetch_and_store(item['path'], depth + 1, max_depth)
                time.sleep(0.5)  # Delay to avoid rate limiting
        elif isinstance(contents, dict) and contents['type'] == 'file' and contents['name'].lower().endswith('.md'):
            save_file(contents, headers)
    else:
        logging.error(f"Failed to list files: {response.status_code}")
        logging.error(f"Error message: {response.text}")

def save_file(file_info, headers):
    global file_counter
    content_response = requests.get(file_info['url'], headers=headers)
    if content_response.status_code == 200:
        content = base64.b64decode(content_response.json()['content']).decode('utf-8')
        file_path = os.path.join(OUTPUT_DIR, file_info['path'])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        file_counter += 1
        logging.info(f"Stored {file_info['path']} successfully. Total files: {file_counter}")
    else:
        logging.error(f"Failed to fetch file content: {content_response.status_code}")

if __name__ == "__main__":
    try:
        fetch_and_store()
    finally:
        logging.info(f"Total Markdown files downloaded: {file_counter}")
        logging.info(f"GITHUB_TOKEN: {'*' * len(ACCESS_TOKEN)}")  # Don't print the actual token
        logging.info(f"REPO_OWNER: {REPO_OWNER}")
        logging.info(f"REPO_NAME: {REPO_NAME}")
        logging.info(f"BRANCH: {BRANCH}")
        logging.info(f"BASE_URL: {BASE_URL}")
        logging.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
