import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
import data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

def load_docs(directory=OUTPUT_DIR):
    if not directory:
        raise ValueError("Directory path is not set")

    logging.info(f"Attempting to load documents from: {directory}")
    try:
        loader = DirectoryLoader(directory, glob="**/*.md", show_progress=True)
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents from {directory}")
        return docs
    except Exception as e:
        logging.error(f"Error loading documents from {directory}: {e}")
        return []

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR) or not os.listdir(OUTPUT_DIR):
        logging.info("No Markdown files found. Starting fetch...")
        data.fetch_and_store_all()
    else:
        logging.info("Markdown files already exist. Skipping fetch...")

    docs = load_docs()
    logging.info(f"Loaded {len(docs)} documents.")