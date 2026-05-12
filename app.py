import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 

# FIX: Corrected import paths from langchain_classic to standard langchain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIG & DARK THEME CSS ---
st.set_page_config(page_title="Lexis Nigeria", page_icon="🇳🇬", layout="wide")

# FIX: Secure API key handling via environment variable instead of direct variable
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.markdown("""
    <style>
    /* Midnight Blue Background */
    .stApp { background-color: #0b1120; color: #f8fafc; }
    h1 { color: #ffffff !important; font-weight: 700; }
    .stCaption { color: #94a3b8 !important; }
    
    /* Force white text for mobile and all chat elements */
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "constitution_db") 

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # FIX: Removed the duplicate ChromaDB initialization
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Cloud Generation (Groq LPU) - Pulls key automatically from os.environ
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant" 
    )
    
    # FIX: Enhanced System Prompt for explicit hallucination prevention
    system_prompt = (
        "You are a legal expert on the 1999 Nigerian Constitution. "
        "Answer the question strictly using the provided context in formal English. "
        "Only respond in Pidgin or Hausa if the user explicitly asks their question in that specific language. "
        "Always maintain a professional and authoritative tone. "
        "If the question cannot be answered from the provided context, say clearly: "
        "'This matter is not addressed in the constitutional sections I have retrieved. Please consult a qualified legal practitioner.'\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), ("human", "{input}"),
    ])
    
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

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
        st.session_state.history = [] # Also clear history array
        st.rerun()
    st.divider()
    for h_query in reversed(st.session_state.history):
        st.caption(f"📜 {h_query[:30]}...")

# --- 5. MAIN CHAT INTERFACE ---
st.title("Lexis Nigeria")
st.caption("Intelligent Constitutional Retrieval Engine • Grounded in the 1999 Constitution")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about your constitutional rights..."):
    # FIX: Allow duplicate questions but cap history at 20 items to prevent UI overflow
    st.session_state.history.append(query)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): 
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Constitution..."):
            # FIX: Added try/except error handling
            try:
                response = rag_pipeline.invoke({"input": query})
                ans = response["answer"]
                st.markdown(ans)
                
                with st.expander("🔍 View Verified Legal Context"):
                    for doc in response["context"]:
                        section_ref = doc.metadata.get("section_ref", "Verified Provision")
                        st.markdown(f"#### 📜 {section_ref}")
                        
                        # FIX: Clean context truncation at the last full stop
                        content = doc.page_content[:500]
                        last_stop = content.rfind('.')
                        display = content[:last_stop + 1] if last_stop > 100 else content
                        
                        st.info(display)
                        st.divider()
                        
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
            except Exception as e:
                error_msg = "I'm having trouble retrieving that information right now. Please check your connection or try again."
                st.error(f"System error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})