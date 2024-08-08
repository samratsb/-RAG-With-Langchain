from langchain_community.document_loaders import DirectoryLoader

DATA_PATH = "markdown_files"  # Update this to your directory path

def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.md", show_progress=True)
    docs = loader.load()
    return docs

# Example usage
docs = load_docs()
print(f"Loaded {len(docs)} documents.")
