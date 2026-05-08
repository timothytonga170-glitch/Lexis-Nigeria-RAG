from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from semantic_chunking import chunk_documents # Imports your successful Phase 3 logic

def build_vector_db():
    # 1. Get the 685 semantic chunks from Phase 3
    print("Fetching semantic units...")
    chunks = chunk_documents()
    # NEW: Metadata Extraction Logic
    import re
    def get_section_label(text):
        match = re.search(r"(Section\s+\d+)", text, re.IGNORECASE)
        return match.group(1) if match else "Constitutional Provision"

    print("Adding metadata citations to chunks...")
    for chunk in chunks:
        chunk.metadata["section_ref"] = get_section_label(chunk.page_content)
    
    # 2. Initialize Embeddings (HuggingFace BGE-Small)
    # This turns text into 384-dimensional math vectors
    print("Initializing BGE Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'} # Optimized for your Dell Latitude
    )

    # 3. Create and Persist ChromaDB
    # This saves the 'knowledge' to your SSD in a folder named 'constitution_db'
    print("Generating vectors and building local database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./constitution_db"
    )
    
    print("\nSUCCESS: Vector database established in './constitution_db'.")
    print("Your RAG system now has a 'brain' grounded in verified legal text.")

if __name__ == "__main__":
    build_vector_db()