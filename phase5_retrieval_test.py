import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def verify_retrieval_logic():
    # 1. Initialize the Embedding Engine (The Math)
    print("--- Phase 5: Initializing Similarity Search Engine ---")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 2. Connect to the Knowledge Base
    db = Chroma(persist_directory="./constitution_db", embedding_function=embeddings)

    # 3. Define the Test Query
    test_query = "What are the fundamental human rights in Nigeria?"
    print(f"User Query: {test_query}")

    # 4. Execute K-Nearest Neighbors (k=3)
    # This is the pure retrieval logic without the LLM
    docs = db.similarity_search(test_query, k=3)

    print(f"\nSUCCESS: Retrieved {len(docs)} relevant legal units.")
    print("-" * 30)

    for i, doc in enumerate(docs):
        print(f"\nNEIGHBOR {i+1} (Source: 1999 Constitution):")
        # Show first 200 characters to verify relevance
        print(f"CONTENT: {doc.page_content[:250]}...")
        print("-" * 15)

if __name__ == "__main__":
    verify_retrieval_logic()