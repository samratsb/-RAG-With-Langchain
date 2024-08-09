from langchain_community.document_loaders import DirectoryLoader #type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings #type: ignore

from langchain_community.embeddings import HuggingFaceEmbeddings #type: ignore
from langchain_chroma import Chroma #type: ignore
from langchain.vectorstores import Chroma # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore

def store_embeddings(chunks):
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')
    
    # Create or connect to a Chroma collection
    client = Chroma()
    collection = client.get_or_create_collection(name='document_embeddings')

    # Generate embeddings for each chunk
    chunk_texts = [chunk['text'] for chunk in chunks]  # Assuming chunks is a list of dicts with 'text' key
    chunk_embeddings = embeddings.embed(chunk_texts)

    # Add embeddings to Chroma
    collection.add(texts=chunk_texts, embeddings=chunk_embeddings)

    print("Embeddings stored successfully.")

def query_embedding(query_text):
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')

    # Create or connect to a Chroma collection
    client = Chroma()
    collection = client.get_or_create_collection(name='document_embeddings')

    # Generate embedding for the query
    query_embedding = embeddings.embed([query_text])

    # Query the collection
    results = collection.query(embedding=query_embedding[0])

    return results
