import logging
import os
import uuid
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Get environment variables
MODEL_NAME = os.getenv('MODEL_NAME')

# Load the model and tokenizer from Hugging Face
logging.info(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize collection variable
collection = None

def initialize_collection():
    global collection
    try:
        # Attempt to create or get the collection
        collection = chroma_client.get_or_create_collection(name="my_embeddings")
        logging.info("Collection initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize collection: {e}")

# Initialize the collection
initialize_collection()

def get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]  # Convert single string to list
    elif isinstance(texts, list):
        texts = [str(text) for text in texts]  # Ensure all elements are strings
    else:
        raise ValueError("Input texts must be a string or a list of strings.")
    
    # Tokenize and encode the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling to get sentence embeddings

    # Convert embeddings to a list of lists for compatibility with ChromaDB
    embeddings = embeddings.cpu().numpy().tolist()
    return embeddings

def get_and_store_embeddings(texts):
    if collection is None:
        logging.error("ChromaDB collection is not initialized. Cannot store embeddings.")
        return None

    embeddings = get_embeddings(texts)
    
    if embeddings:
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Store embeddings in ChromaDB
        try:
            collection.add(
                texts=texts,
                embeddings=embeddings,
                ids=ids
            )
            logging.info(f"Stored {len(embeddings)} embeddings in ChromaDB")
        except Exception as e:
            logging.error(f"Failed to store embeddings: {e}")
            return None
        
        return embeddings
    else:
        logging.error("Failed to get embeddings")
        return None

def query_embeddings(query_text):
    if collection is None:
        logging.error("ChromaDB collection is not initialized. Cannot query embeddings.")
        return None

    query_embedding = get_embeddings([query_text])
    
    if query_embedding:
        try:
            # Query embeddings from ChromaDB
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3  # Adjust the number of results as needed
            )
            return results
        except Exception as e:
            logging.error(f"Failed to query embeddings: {e}")
            return None
    else:
        logging.error("Failed to get query embedding")
        return None

def main():
    while True:
        action = input("Enter 'add' to add text, 'query' to query text, or 'exit' to exit: ").strip().lower()

        if action == 'add':
            text = input("Enter text to add: ").strip()
            if text:
                get_and_store_embeddings([text])
                print("Text added successfully.")
            else:
                print("No text provided.")

        elif action == 'query':
            query_text = input("Enter text to query: ").strip()
            if query_text:
                results = query_embeddings(query_text)
                if results:
                    print("Query results:", results)
                else:
                    print("No results found.")
            else:
                print("No query text provided.")

        elif action == 'exit':
            print("Exiting.")
            break

        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
