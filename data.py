import requests, base64, os #type: ignore
from dotenv import load_dotenv #type: ignore

load_dotenv()

#loaded from .env file
ACCESS_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = os.getenv('REPO_OWNER')
REPO_NAME = os.getenv('REPO_NAME')
BRANCH = os.getenv('BRANCH')

BASE_URL = os.getenv('BASE_URL')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

#check if you have created the directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_and_store(path=''):
    url = f'{BASE_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}'
    headers = {'Authorization': f'token {ACCESS_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers, params={'ref': BRANCH})

    if response.status_code == 200:
        for file in response.json():
            if file['name'].lower().endswith('.md'):
                content_response = requests.get(file['url'], headers=headers)
                if content_response.status_code == 200:
                    content = base64.b64decode(content_response.json()['content']).decode('utf-8')
                    with open(os.path.join(OUTPUT_DIR, file['name']), 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Stored {file['name']} successfully.")
                else:
                    print(f"Failed to fetch file content: {content_response.status_code}")
    else:
        print(f"Failed to list files: {response.status_code}")

fetch_and_store()