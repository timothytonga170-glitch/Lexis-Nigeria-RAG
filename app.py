import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIG & DARK THEME CSS ---
st.set_page_config(page_title="Lexis Nigeria", page_icon="🇳🇬", layout="wide")

# Your Active Groq API Key
GROQ_API_KEY = "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"

st.markdown("""
    <style>
    /* Midnight Blue Background */
    .stApp { background-color: #0b1120; color: #f8fafc; }
    h1 { color: #ffffff !important; font-weight: 700; }
    .stCaption { color: #94a3b8 !important; }
    
    /* FIX: Force white text for mobile and all chat elements */
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] li, 
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #ffffff !important;
    }

    /* User Message Style (Right) */
    [data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        flex-direction: row-reverse; text-align: right; background-color: #1e293b;
        border: 1px solid #334155; border-radius: 20px; margin-left: auto; max-width: 75%; padding: 10px;
    }
    
    /* Assistant Message Style (Left) */
    [data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
        background-color: #1e293b; border: 1px solid #008751; border-radius: 20px;
        max-width: 85%; margin-bottom: 25px;
    }
    
    /* Sidebar & Footer Fixes */
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    footer {visibility: hidden;}
    .stChatInputContainer { padding-bottom: 20px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE (HYBRID RAG) ---
@st.cache_resource
def setup_engine():
    # Phase 5: Local Retrieval with Absolute Path Resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # UPDATED: Point directly to the folder with your 685 semantic chunks
    db_path = os.path.join(current_dir, "constitution_db") 

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Initialize Vector Store
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Cloud Generation (Groq LPU)
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name="llama-3.1-8b-instant" 
    )
    
    system_prompt = (
        "You are a legal expert on the 1999 Nigerian Constitution. "
        "Answer the question strictly using the provided context in formal English. "
        "Only respond in Pidgin or Hausa if the user explicitly asks their question in that specific language. "
        "Always maintain a professional and authoritative tone."
        "\n\nContext: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), ("human", "{input}"),
    ])
    
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# Initialize the RAG Pipeline
rag_pipeline = setup_engine()

# --- 3. SESSION STATE ---
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "history" not in st.session_state: 
    st.session_state.history = []

# --- 4. SIDEBAR HISTORY ---
with st.sidebar:
    st.markdown("### 🏛 **Legal History**")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    for h_query in reversed(st.session_state.history):
        st.caption(f"📜 {h_query[:30]}...")

# --- 5. MAIN CHAT INTERFACE ---
st.title("Lexis Nigeria")
st.caption("Intelligent Constitutional Retrieval Engine • Grounded in the 1999 Constitution")

# Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Logic
if query := st.chat_input("Ask about your constitutional rights..."):
    # Save to sidebar history
    if query not in st.session_state.history: 
        st.session_state.history.append(query)
    
    # Display User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): 
        st.markdown(query)

    # Generate and display Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Searching the Constitution..."):
            response = rag_pipeline.invoke({"input": query})
            ans = response["answer"]
            st.markdown(ans)
            
            # Display Metadata "Section" citations
            with st.expander("🔍 View Verified Legal Context"):
                for doc in response["context"]:
                    # Pull the label created in the build script
                    section_ref = doc.metadata.get("section_ref", "Verified Provision")
                    st.markdown(f"#### 📜 {section_ref}")
                    st.info(f"{doc.page_content[:450]}...")
                    st.divider()
    
    st.session_state.messages.append({"role": "assistant", "content": ans})