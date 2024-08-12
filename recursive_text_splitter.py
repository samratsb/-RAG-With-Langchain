import logging
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import document_loader as dl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_FILE = 'chunks_cache.pkl'

def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced chunk size for testing
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = []

    for doc in docs:
        try:
            split_texts = splitter.split_text(doc.page_content)
            logging.info(f"Document split into {len(split_texts)} chunks.")
            for i, chunk in enumerate(split_texts):
                chunks.append({"page_content": chunk, "metadata": {**doc.metadata, "chunk_index": i}})
        except Exception as e:
            logging.error(f"Error splitting document: {e}")

    return chunks

def cache_chunks(chunks):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(chunks, f)
    logging.info(f"Cached {len(chunks)} chunks to {CACHE_FILE}")

def load_cached_chunks():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            chunks = pickle.load(f)
        logging.info(f"Loaded {len(chunks)} chunks from cache")
        return chunks
    return None

if __name__ == "__main__":
    cached_chunks = load_cached_chunks()
    
    if cached_chunks:
        chunks = cached_chunks
    else:
        docs = dl.load_docs()
        if docs:
            logging.info(f"Loaded {len(docs)} documents. Starting to split...")
            chunks = split_text(docs)
            cache_chunks(chunks)
        else:
            logging.warning("No documents found to split.")
            chunks = []

    logging.info(f"Working with a total of {len(chunks)} chunks across all documents.")
    
    if chunks:
        logging.info(f"Sample chunk content (first 100 characters): {chunks[0]['page_content'][:100]}")
        if len(chunks) > 1:
            logging.info(f"Sample chunk content (last chunk, first 100 characters): {chunks[-1]['page_content'][:100]}")