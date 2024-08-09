import document_loader as dl
from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore

def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = []

    for doc in docs:
        text = getattr(doc, "page_content", None) or doc
        if isinstance(text, str):
            try:
                chunks.extend(splitter.split_text(text))
            except Exception as e:
                print(f"Error splitting document: {e}")
        else:
            print(f"Document is not a string or does not have a page_content attribute: {doc}")
    return chunks

if __name__ == "__main__":
    import document_loader as dl
    docs = dl.load_docs()
    chunks = split_text(docs)
    print(f"Created {len(chunks)} chunks.")