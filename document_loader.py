import os
import logging
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
import data
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

def find_markdown_files(directory):
    if not directory:
        raise ValueError("Directory path is not set")

    markdown_files = []
    for root, dirs, files in os.walk(directory):
        logging.info(f"Checking directory: {root}")
        for file in files:
            if file.lower().endswith('.md'):
                file_path = os.path.join(root, file)
                logging.info(f"Found markdown file: {file} in {root}")
                markdown_files.append(file_path)
    return markdown_files

def load_docs(file_paths):
    logging.info(f"Attempting to load documents from: {file_paths}")
    docs = []
    for file_path in file_paths:
        try:
            loader = TextLoader(file_path)
            docs.extend(loader.load())
            logging.info(f"Loaded document from: {file_path}")
        except Exception as e:
            logging.error(f"Error loading document from {file_path}: {e}")
    logging.info(f"Total documents loaded: {len(docs)}")
    return docs

if __name__ == "__main__":
    markdown_files = find_markdown_files(OUTPUT_DIR)

    if not markdown_files:
        logging.info("No Markdown files found. Starting fetch...")
        data.fetch_and_store()
        markdown_files = find_markdown_files(OUTPUT_DIR)
    else:
        logging.info("Markdown files already exist. Skipping fetch...")

    docs = load_docs(markdown_files)
    logging.info(f"Loaded {len(docs)} documents.")
