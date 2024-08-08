import document_loader as dl
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))
    return chunks

# Example usage
chunks = split_text(dl.docs)
print(f"Created {len(chunks)} chunks.")
