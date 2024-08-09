from langchain.document_loaders import DirectoryLoader #type: ignore
import data as data

def load_docs():
    loader = DirectoryLoader("markdown_files", glob="*.md", show_progress=True, use_multithreading=True)
    return loader.load()

if __name__ == "__main__":
    data.fetch_and_store()
    docs = load_docs()
    print(f"Loaded {len(docs)} documents.")