import logging
import time
import requests
from dotenv import load_dotenv
import os
import chromadb
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Get environment variables
MODEL_NAME = os.getenv('MODEL_NAME')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

# Define the Hugging Face API URL
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Set up headers with API token
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection
try:
    collection = chroma_client.get_or_create_collection(name="my_embeddings")
    logging.info("Collection initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize collection: {e}")

def get_embeddings(texts):
    payload = {"inputs": texts}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 503:
        error_info = response.json()
        estimated_time = error_info.get("estimated_time", 10)
        logging.error(f"Model is loading. Retrying in {estimated_time} seconds...")
        time.sleep(estimated_time)
        return get_embeddings(texts)
    else:
        logging.error(f"Failed to get embeddings: {response.status_code}")
        logging.error(f"Response: {response.text}")
        return None

def get_and_store_embeddings(texts):
    embeddings_response = get_embeddings(texts)
    
    if embeddings_response:
        embeddings = embeddings_response.get("embeddings", [])
        
        if not embeddings:
            logging.error("No embeddings returned from API")
            return None
        
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Store embeddings in ChromaDB
        collection.add(
            texts=texts,
            embeddings=embeddings,
            ids=ids
        )
        
        logging.info(f"Stored {len(embeddings)} embeddings in ChromaDB")
        return embeddings
    else:
        logging.error("Failed to get embeddings")
        return None

def query_embeddings(query_text):
    query_embedding_response = get_embeddings([query_text])
    
    if query_embedding_response:
        query_embedding = query_embedding_response.get("embeddings", [])
        
        if not query_embedding:
            logging.error("No query embedding returned from API")
            return None
        
        # Query embeddings from ChromaDB
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=10  # Adjust the number of results as needed
        )
        return results
    else:
        logging.error("Failed to get query embedding")
        return None

if __name__ == "__main__":
    # Example usage
    texts = ["Sample text for embedding."]
    get_and_store_embeddings(texts)
    results = query_embeddings("Sample text for query.")
    print("Query results:", results)
