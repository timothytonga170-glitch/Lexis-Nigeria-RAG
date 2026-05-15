import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 2026 Updated LangChain Imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setting up the look and feel of the app
st.set_page_config(page_title="Lexis Nigeria", page_icon="NG", layout="wide")

# Securely loading the API key via Streamlit Secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.markdown("""
    <style>
    /* Midnight Blue Background */
    .stApp { background-color: #0b1120; color: #f8fafc; }
    h1 { color: #ffffff !important; font-weight: 700; }
    .stCaption { color: #94a3b8 !important; }
    
    /* White text for all chat elements */
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
    
    /* Sidebar and Footer Fixes */
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    footer {visibility: hidden;}
    .stChatInputContainer { padding-bottom: 20px !important; }
    </style>
    """, unsafe_allow_html=True)

# The brain of the application
@st.cache_resource
def setup_engine():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "constitution_db") 

        # OPTIMAL: Using 'base' instead of 'small' for better legal terminology capture
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # OPTIMAL: Increased k to 5 to ensure overlapping provisions (like President vs Governor) are caught
        retriever = db.as_retriever(search_kwargs={"k": 5})
        
        llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.1-8b-instant" 
        )
        
        # OPTIMAL: Refined System Prompt to prevent hallucination and mix-ups
        system_prompt = (
            "You are a high-precision legal expert on the 1999 Nigerian Constitution. "
            "Use the provided context to answer the question. "
            "STRICT RULES:\n"
            "1. Answer strictly using the provided context.\n"
            "2. If the context compares different offices (e.g., President vs Governor), clearly distinguish between them.\n"
            "3. If the answer is not in the context, say: 'This specific matter is not addressed in the retrieved constitutional sections. Please consult a legal practitioner.'\n"
            "4. Maintain an authoritative, professional tone.\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), ("human", "{input}"),
        ])
        
        return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        
    except Exception as e:
        st.error(f"Critical System Error: Failed to load the AI Engine. Details: {str(e)}")
        return None

rag_pipeline = setup_engine()

# Session state management
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "history" not in st.session_state: 
    st.session_state.history = []

# Sidebar for the chat history
with st.sidebar:
    st.markdown("### 🔍 **Legal History**")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.history = [] 
        st.rerun()
    st.divider()
    for h_query in reversed(st.session_state.history):
        st.caption(f"📌 {h_query[:30]}...")

# Main interface
st.title("Lexis Nigeria")
st.caption("Intelligent Constitutional Retrieval Engine • Grounded in the 1999 Constitution")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about your constitutional rights..."):
    st.session_state.history.append(query)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): 
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Constitution..."):
            if rag_pipeline is None:
                st.error("The retrieval engine is currently offline.")
            else:
                try:
                    response = rag_pipeline.invoke({"input": query})
                    ans = response["answer"]
                    st.markdown(ans)
                    
                    with st.expander("📖 View Verified Legal Context"):
                        for doc in response.get("context", []):
                            section_ref = doc.metadata.get("section_ref", "Verified Provision")
                            st.markdown(f"#### 📜 {section_ref}")
                            st.info(f"{doc.page_content[:600]}...") # Increased character count for better readability
                            st.divider()
            
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    
                except Exception as e:
                    st.error("Issue processing request with AI provider. Please try again.")
                    with st.expander("Technical Error Details"):
                        st.write(e)