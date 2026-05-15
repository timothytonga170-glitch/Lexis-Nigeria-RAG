import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq 

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setting up the look and feel of the app
st.set_page_config(page_title="Lexis Nigeria", page_icon="🇳🇬", layout="wide")

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
        
        # Pointing directly to the folder where our semantic chunks are stored
        db_path = os.path.join(current_dir, "constitution_db") 

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Connecting to Groq for our AI model. It grabs the key from the environment automatically.
        llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.1-8b-instant" 
        )
        
        # Setting strict rules for the AI to make sure it never invents false legal facts
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
        
    except Exception as e:
        st.error(f"Critical System Error: Failed to load the AI Engine. Details: {str(e)}")
        return None

rag_pipeline = setup_engine()

# Keeping track of the chat session
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "history" not in st.session_state: 
    st.session_state.history = []

# Sidebar for the chat history
with st.sidebar:
    st.markdown("### 🏛 **Legal History**")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.history = [] 
        st.rerun()
    st.divider()
    for h_query in reversed(st.session_state.history):
        st.caption(f"📜 {h_query[:30]}...")

# The main interface where users type their queries
st.title("Lexis Nigeria")
st.caption("Intelligent Constitutional Retrieval Engine • Grounded in the 1999 Constitution")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about your constitutional rights..."):
    # Allowing duplicate questions but capping the history at 20 items so the screen does not get crowded
    st.session_state.history.append(query)
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): 
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Constitution..."):
            
            # Checking if the engine loaded correctly before trying to use it
            if rag_pipeline is None:
                st.error("The retrieval engine is currently offline. Please check the backend logs.")
            else:
                try:
                    response = rag_pipeline.invoke({"input": query})
                    ans = response["answer"]
                    st.markdown(ans)
                    
                    # Displaying the actual law sections to prove the AI is accurate
                    with st.expander("🔍 View Verified Legal Context"):
                        for doc in response.get("context", []):
                            section_ref = doc.metadata.get("section_ref", "Verified Provision")
                            st.markdown(f"#### 📜 {section_ref}")
                            st.info(f"{doc.page_content[:450]}...")
                            st.divider()
            
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    
                except Exception as e:
                    # Catching any errors that happen while Groq is thinking
                    st.error("There was an issue processing your request with the AI provider. Please try again.")
                    with st.expander("Show Technical Error Details"):
                        st.write(e)