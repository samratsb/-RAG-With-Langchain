import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get model name and Hugging Face API token from environment variables
MODEL_NAME = os.getenv('MODEL_NAME')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')  # Ensure this is set in your .env file

# Define the Hugging Face API URL
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# Set up headers with API token
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def get_embeddings(texts):
    # Make a POST request to the Hugging Face API
    response = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": texts})
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get embeddings: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def store_embeddings(chunks):
    # Ensure chunks are in the expected format
    if isinstance(chunks[0], str):
        chunk_texts = chunks
    else:
        chunk_texts = [chunk.get('text', '') for chunk in chunks]

    # Get embeddings for each chunk
    embeddings_response = get_embeddings(chunk_texts)
    
    if embeddings_response:
        # Process embeddings as needed
        chunk_embeddings = embeddings_response["embeddings"]
        
        # Create or connect to a Chroma collection
        client = Chroma()
        collection = client.get_or_create_collection(name=MODEL_NAME)

        # Add embeddings to Chroma
        collection.add(texts=chunk_texts, embeddings=chunk_embeddings)

        print("Embeddings stored successfully.")
    else:
        print("Failed to store embeddings.")

def query_embedding(query_text):
    # Get embedding for the query
    query_embedding_response = get_embeddings([query_text])
    
    if query_embedding_response:
        query_embedding = query_embedding_response["embeddings"][0]
        
        # Create or connect to a Chroma collection
        client = Chroma()
        collection = client.get_or_create_collection(name=MODEL_NAME)

        # Query the collection
        results = collection.query(embedding=query_embedding)

        return results
    else:
        print("Failed to query embedding.")
        return None
