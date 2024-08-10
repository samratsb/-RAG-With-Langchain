import document_loader as dl
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdb

def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        length_function=len,
        add_start_index=True,
    )
    chunks = []

    for doc in docs:
        # Handle different document structures
        if hasattr(doc, "page_content"):
            text = getattr(doc, "page_content", "")
        elif isinstance(doc, str):
            text = doc
        else:
            print(f"Skipping document: {doc}")
            continue

        if isinstance(text, str):
            try:
                split_texts = splitter.split_text(text)
                for i, chunk in enumerate(split_texts):
                    # Create a document with page_content
                    chunks.append({"page_content": chunk, "metadata": {"index": i}})
            except Exception as e:
                print(f"Error splitting document: {e}")
        else:
            print(f"Document content is not a string: {text}")

    return chunks

if __name__ == "__main__":
    docs = dl.load_docs()
    if docs:
        chunks = split_text(docs)
        print(f"Created {len(chunks)} chunks.")
    else:
        print("No documents found to split.")
