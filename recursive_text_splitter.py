import document_loader as dl
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(docs):
    # Initialize the RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = []

    # Iterate over each document and split it into chunks
    for doc in docs:
        # Check if doc is a Document object with page_content
        text = getattr(doc, "page_content", None) or doc
        if isinstance(text, str):
            try:
                chunks.extend(splitter.split_text(text))
            except Exception as e:
                print(f"Error splitting document: {e}")
        else:
            print(f"Document is not a string or does not have a page_content attribute: {doc}")
    return chunks

# Example usage
docs = dl.load_docs()  # Ensure dl.docs is a list of documents
chunks = split_text(docs)
print(f"Created {len(chunks)} chunks.")
