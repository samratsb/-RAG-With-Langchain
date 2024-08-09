from data import fetch_and_store
from document_loader import load_docs
from recursive_text_splitter import split_text
from embeddings import store_embeddings, query_embedding


def main():
    # Fetch,store and load documents from GitHub
    docs = load_docs()
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