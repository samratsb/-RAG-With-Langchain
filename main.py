from document_loader import load_docs
from recursive_text_splitter import split_text
from embeddings import store_embeddings, query_embedding
import os
from dotenv import load_dotenv

load_dotenv()
OUTPUT_DIR = os.getenv('OUTPUT_DIR')

def main():
    # Find markdown files in the directory
    from document_loader import find_markdown_files
    markdown_files = find_markdown_files(OUTPUT_DIR)

    if not markdown_files:
        print("No Markdown files found. Please ensure documents are fetched.")
        return

    # Load documents
    docs = load_docs(markdown_files)
    print(f"Loaded {len(docs)} documents.")

    # Split documents into chunks
    chunks = split_text(docs)
    print(f"Created {len(chunks)} chunks.")

    # Create and store embeddings
    store_embeddings(chunks)

    # Example query
    while True:
        query_text = input("Enter your query (or type 'exit' to quit): ")
        if query_text.lower() == 'exit':
            break
        results = query_embedding(query_text)
        print("Query results:", results)

if __name__ == "__main__":
    main()
