import streamlit as st
import os
import tempfile
import shutil
import chromadb
import logging
from typing import Optional, List

# LangChain & AI Modules
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# =========================
# Configuration & Setup
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DevAssist AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {max-width: 1200px; margin: 0 auto;}
    .metric-card {
        background: white; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Constants
# =========================

MODEL_CONFIG = {
    "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 400,
    "chunk_overlap": 50,
    "retrieval_k": 3
}

DEV_KEYWORDS = [
    "project", "document", "file", "system", "architecture", "stack",
    "prd", "assignment", "spec", "design", "implementation", "technology",
    "requirements", "features", "summary", "codebase", "frontend", "backend",
    "api", "database", "process", "workflow", "deployment", "code", "module",
    "component", "framework", "library", "service", "endpoint", "authentication"
]

# =========================
# Helper Functions
# =========================

def create_directories():
    for dir_name in ["uploads", "data", "temp"]:
        os.makedirs(dir_name, exist_ok=True)

def is_dev_question(query: str) -> bool:
    query_lower = query.lower()
    if any(k in query_lower for k in DEV_KEYWORDS):
        return True
    question_phrases = ["what is", "how to", "explain", "describe", "summary of", "tell me about", "list", "show me"]
    return any(p in query_lower for p in question_phrases)

def format_answer(answer: str) -> str:
    lines = [l.strip() for l in answer.split('\n') if l.strip()]
    seen, unique = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return '\n'.join(unique)

# =========================
# Model Loading
# =========================

@st.cache_resource
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name=MODEL_CONFIG["embeddings_model"])
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

@st.cache_resource
def load_llm():
    try:
        return ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0
        )
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

# =========================
# Document Processing
# =========================

def process_document(uploaded_file) -> Optional[RetrievalQA]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        documents = [d for d in documents if d.page_content.strip()]

        if not documents:
            st.error("No valid content found in this PDF.")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MODEL_CONFIG["chunk_size"],
            chunk_overlap=MODEL_CONFIG["chunk_overlap"]
        )
        chunks = splitter.split_documents(documents)
        st.sidebar.info(f"📑 Created {len(chunks)} text chunks")

        embeddings = load_embeddings()
        if not embeddings:
            return None

        client = chromadb.PersistentClient(path="./data")
        try:
            client.delete_collection("active_document")
        except:
            pass

        vectordb = Chroma(
            persist_directory="./data",
            embedding_function=embeddings,
            collection_name="active_document"
        )
        vectordb.add_documents(chunks)

        retriever = vectordb.as_retriever(search_kwargs={"k": MODEL_CONFIG["retrieval_k"]})
        llm = load_llm()
        if not llm:
            return None

        prompt_template = """You are DevAssist AI, a specialized assistant for developer documentation.

INSTRUCTIONS:
1. Answer ONLY using the provided context below.
2. If the answer is not in the context, reply exactly: "❌ I could not find this information in the uploaded document."
3. Be accurate, concise, and specific. Do not make up information.
4. Use bullet points where helpful.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        os.unlink(file_path)
        return qa_chain

    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

# =========================
# UI Components
# =========================

def render_sidebar():
    st.sidebar.title("📂 Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload Developer PDF", type=["pdf"])
    dev_mode = st.sidebar.toggle("🧠 Developer Mode", value=False)
    return uploaded_file, dev_mode

def render_chat_history(chat_history, dev_mode):
    for q, a, sources in chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if sources and dev_mode:
                st.markdown("**📚 Sources:**")
                seen = set()
                for doc in sources:
                    source = os.path.basename(doc.metadata.get("source", "Document"))
                    page = doc.metadata.get("page", "N/A")
                    key = (source, page)
                    if key not in seen:
                        seen.add(key)
                        st.markdown(f"• {source} (page {page})")

# =========================
# Main Application Logic
# =========================

def main():
    create_directories()

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = None

    uploaded_file, dev_mode = render_sidebar()

    st.title("🚀 DevAssist AI")
    st.markdown("### Your Developer Documentation Assistant")

    if not uploaded_file:
        st.info("👋 Upload a PDF (PRD, API Doc, Spec, etc.) to start.")
        return

    if uploaded_file and st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.sidebar.info("Processing new document...")
        qa_chain = process_document(uploaded_file)
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.sidebar.success("✅ Document processed successfully!")
        else:
            st.sidebar.error("Failed to process document.")
            return

    query = st.chat_input("Ask something about your document...")

    if query:
        if not is_dev_question(query):
            answer = "❌ This question doesn't seem related to developer documentation."
            sources = []
        else:
            with st.spinner("🔍 Searching the document..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": query})
                    answer = format_answer(result["result"])
                    sources = result.get("source_documents", [])
                except Exception as e:
                    answer = "❌ Error getting answer. Please try again."
                    sources = []
                    st.error(str(e))
        st.session_state.chat_history.append((query, answer, sources))
        st.rerun()

    if st.session_state.chat_history:
        render_chat_history(st.session_state.chat_history, dev_mode)

    st.markdown("---")
    st.caption("*⚙️ DevAssist AI — Built with Streamlit, LangChain, and Groq*")

if __name__ == "__main__":
    main()