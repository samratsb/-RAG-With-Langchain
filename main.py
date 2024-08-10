from document_loader import load_docs, find_markdown_files
from recursive_text_splitter import split_text
from embeddings import get_embeddings, get_and_store_embeddings, query_embeddings
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

def main():
    if not OUTPUT_DIR:
        logging.error("OUTPUT_DIR not set in .env file")
        return

    # Find markdown files in the directory
    markdown_files = find_markdown_files(OUTPUT_DIR)

    if not markdown_files:
        logging.warning("No Markdown files found. Please ensure documents are fetched.")
        return

    # Load documents
    docs = load_docs(markdown_files)
    logging.info(f"Loaded {len(docs)} documents.")

    # Split documents into chunks
    chunks = split_text(docs)
    logging.info(f"Created {len(chunks)} chunks.")

    # Create and store embeddings
    embeddings = get_and_store_embeddings(chunks)
    if not embeddings:
        logging.error("Failed to create and store embeddings. Exiting.")
        return

    # Example query
    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ")
        if query_text.lower() == 'exit':
            break
        results = query_embeddings(query_text)
        if results:
            logging.info("Query results:")
            for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
                logging.info(f"{i}. Document: {doc[:100]}... (Score: {score})")
        else:
            logging.warning("No results found for the query.")

if __name__ == "__main__":
    main()