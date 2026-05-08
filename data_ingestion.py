import os
import re
from langchain_community.document_loaders import PyPDFLoader

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
    run_ingestion()