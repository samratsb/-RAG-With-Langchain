# Fetch markdown files from certain repos and store them locally in the markdown_files folder

import requests # type: ignore
import base64
import os
from dotenv import load_dotenv #type: ignore

load_dotenv()

ACCESS_TOKEN = os.getenv('GITHUB_TOKEN')  # Add your own access token here

REPO_OWNER = 'pytorch'
REPO_NAME = 'pytorch'
BRANCH = 'main'

BASE_URL = 'https://api.github.com'

OUTPUT_DIR = 'markdown_files'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_markdown_files(repo_owner, repo_name, branch='main', path=''):
    url = f'{BASE_URL}/repos/{repo_owner}/{repo_name}/contents/{path}'
    headers = {
        'Authorization': f'token {ACCESS_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }
    params = {
        'ref': branch
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file['name'].lower().endswith('.md'):
                file_path = file['path']
                fetch_and_store_file_content(file_path)
    else:
        print(f"Failed to list files: {response.status_code}")
        print(response.json())

def fetch_and_store_file_content(file_path):
    url = f'{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}'
    headers = {
        'Authorization': f'token {ACCESS_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        content = base64.b64decode(file_info['content']).decode('utf-8')

        # Write the content to a local file
        local_file_path = os.path.join(OUTPUT_DIR, file_info['name'])
        with open(local_file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Stored {file_info['name']} successfully.")
    else:
        print(f"Failed to fetch file content: {response.status_code}")
        print(response.json())

# Fetch and store markdown files
fetch_markdown_files(REPO_OWNER, REPO_NAME, branch=BRANCH)
