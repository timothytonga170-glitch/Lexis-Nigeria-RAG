from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_ingestion import run_ingestion # Pulling from your Phase 2 script

def chunk_documents():
    # Step 1: Load the cleaned pages from Phase 2
    print("Loading cleaned constitutional documents...")
    cleaned_pages = run_ingestion("data")
    
    if not cleaned_pages:
        print("No documents found. Please check your data folder.")
        return []

    # Step 2: Configure Parameters & Initialize Splitter
    print("\nInitializing the RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        # These separators tell the splitter to prioritize breaking at paragraphs first, then sentences, then words.
        separators=["\n\n", "\n", ".", " ", ""] 
    )
    
    # Step 3: Execute the Split
    print("Executing semantic chunking...")
    semantic_chunks = text_splitter.split_documents(cleaned_pages)
    
    print(f"\nSUCCESS: Divided the constitution into {len(semantic_chunks)} semantic units.")
    
    # Verification: Print a sample chunk to ensure sentences are not cut in half
    if len(semantic_chunks) > 50:
        print("\n--- SAMPLE CHUNK PREVIEW ---")
        print(semantic_chunks[50].page_content)
        print("----------------------------\n")
        
    return semantic_chunks

if __name__ == "__main__":
    chunks = chunk_documents()