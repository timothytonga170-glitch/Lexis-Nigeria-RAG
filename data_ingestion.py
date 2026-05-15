import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def clean_text(text):
    # Rule 1: Remove vertical sidebar text found in PLAC PDFs
    sidebar = r"The Constitution of the Federal Republic of Nigeria Updated with the First, Second, Third, Fourth and Fifth Alterations"
    text = re.sub(sidebar, "", text)
    
    # Rule 2: Remove PLAC specific contact/address watermarks
    plac_info = r"Policy and Legal Advocacy Centre \(PLAC\).*?Abuja\."
    text = re.sub(plac_info, "", text, flags=re.DOTALL)
    
    # Rule 3: Remove standalone page numbers (digits on their own line)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    
    # Rule 4: Clean up excessive whitespace and newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def run_ingestion(folder_path="data"):
    all_pages = []
    print(f"Scanning the '{folder_path}' folder for PDFs...")
    
    # Verify the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found. Please create it.")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            file_path = os.path.join(folder_path, filename)
            
            # Load the PDF page by page
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Clean each page
            for page in pages:
                page.page_content = clean_text(page.page_content)
                all_pages.append(page)
                
    print(f"Phase 2 Complete: {len(all_pages)} pages cleaned and ready.")
    return all_pages

if __name__ == "__main__":
    # Execute Phase 1 & 2
    documents = run_ingestion()
    
    if documents:
        # Phase 3: Chunking
        print("Phase 3: Chunking the legal text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} semantic chunks.")

        # Phase 4: Building the Vector Database
        print("Phase 4: Building the Vector Database...")
        # Using the base model to match the optimal app.py settings
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # Pointing explicitly to the right path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "constitution_db")
        
        # Creating and saving the database
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )
        
        print("SUCCESS: 'constitution_db' folder has been fully generated!")