import streamlit as st
import os
import tempfile
import shutil
from typing import List, Tuple, Optional

# Document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ML Models
from transformers import pipeline

# Utilities
import time
import logging

# =========================
# Configuration & Setup
# =========================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="DevAssist AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Constants
# =========================

MODEL_CONFIG = {
    "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "google/flan-t5-base",
    "max_new_tokens": 150,
    "chunk_size": 400,
    "chunk_overlap": 50,
    "retrieval_k": 3
}

DEV_KEYWORDS = [
    "project", "document", "file", "system", "architecture", "stack", 
    "prd", "assignment", "spec", "design", "implementation", "technology", 
    "tech", "tools", "requirements", "features", "summary", "codebase", 
    "flow", "frontend", "backend", "ui", "api", "ai", "agent", "database", 
    "process", "workflow", "deployment", "code", "module", "component",
    "framework", "library", "service", "endpoint", "authentication"
]

# =========================
# Helper Functions
# =========================

def create_directories():
    """Create necessary directories for the app"""
    for dir_name in ["uploads", "data", "temp"]:
        os.makedirs(dir_name, exist_ok=True)

def clean_temp_files():
    """Clean up temporary files"""
    try:
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        os.makedirs("temp", exist_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")

def is_dev_question(query: str) -> bool:
    """Enhanced question classification"""
    query_lower = query.lower()
    
    # Check for direct matches
    if any(keyword in query_lower for keyword in DEV_KEYWORDS):
        return True
    
    # Check for question patterns
    dev_patterns = [
        "what is", "how to", "explain", "describe", "summary of",
        "tell me about", "what are the", "list", "show me"
    ]
    
    return any(pattern in query_lower for pattern in dev_patterns)

def format_answer(answer: str) -> str:
    """Clean and format the answer"""
    lines = [line.strip() for line in answer.split('\n') if line.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    
    return '\n'.join(unique_lines)

# =========================
# Core Functions
# =========================

@st.cache_resource
def load_embeddings():
    """Load and cache embeddings model"""
    try:
        return HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embeddings_model"],
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

@st.cache_resource
def load_llm():
    try:
        pipe = pipeline(
            "text-generation",
            model=MODEL_CONFIG["llm_model"],
            max_new_tokens=MODEL_CONFIG["max_new_tokens"],
            do_sample=False
        )

        return HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None


def process_document(uploaded_file) -> Optional[RetrievalQA]:
    """Process uploaded document and create QA chain"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name

        # Load document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Filter empty documents
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        if not documents:
            st.error("No content found in the PDF file")
            return None

        # Update metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MODEL_CONFIG["chunk_size"],
            chunk_overlap=MODEL_CONFIG["chunk_overlap"]
        )
        chunks = splitter.split_documents(documents)
        
        st.sidebar.info(f"Created {len(chunks)} text chunks")

        # Load embeddings
        embeddings = load_embeddings()
        if not embeddings:
            return None

        # Create vector store
        collection_name = "active_document"
        
        # Clean existing collection
        try:
            temp_db = Chroma(
                persist_directory="./data",
                embedding_function=embeddings,
                collection_name=collection_name
            )
            temp_db._client.delete_collection(name=collection_name)
        except:
            pass

        # Create new vector store
        vectordb = Chroma(
            persist_directory="./data",
            embedding_function=embeddings,
            collection_name=collection_name
        )
        vectordb.add_documents(chunks)

        # Create retriever
        retriever = vectordb.as_retriever(
            search_kwargs={"k": MODEL_CONFIG["retrieval_k"]}
        )

        # Load LLM
        llm = load_llm()
        if not llm:
            return None

        # Create prompt template
        prompt_template = """You are DevAssist AI, a specialized assistant for developer documentation.

INSTRUCTIONS:
1. Answer ONLY using the provided context
2. If the answer is not in the context, respond exactly: "❌ I could not find this information in the uploaded document."
3. Keep answers concise and well-structured
4. Use bullet points for lists
5. Focus on technical accuracy

Context:
{context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Cleanup
        os.unlink(file_path)
        
        return qa_chain

    except Exception as e:
        st.error(f"Error processing document: {e}")
        logger.error(f"Document processing error: {e}")
        return None

def stream_text(text: str, placeholder):
    """Create a streaming text effect"""
    streamed_text = ""
    words = text.split()
    
    for i, word in enumerate(words):
        streamed_text += word + " "
        placeholder.write(streamed_text)
        if i % 3 == 0:  # Add slight delay every few words
            time.sleep(0.05)

# =========================
# Session State Initialization
# =========================

def initialize_session_state():
    """Initialize all session state variables"""
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}
    
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

# =========================
# UI Components
# =========================

def render_sidebar():
    """Render the sidebar with upload and controls"""
    st.sidebar.title("📁 Document Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Developer Documentation",
        type=["pdf"],
        help="Supports PRDs, Technical Specs, API Docs, etc."
    )
    
    # Developer mode toggle
    dev_mode = st.sidebar.toggle(
        "🧠 Developer Mode", 
        value=False,
        help="Show debugging information and retrieval details"
    )
    
    # Settings section
    with st.sidebar.expander("⚙️ Settings"):
        st.info("**Current Configuration:**")
        st.text(f"• Embeddings: {MODEL_CONFIG['embeddings_model'].split('/')[-1]}")
        st.text(f"• LLM: {MODEL_CONFIG['llm_model'].split('/')[-1]}")
        st.text(f"• Chunk Size: {MODEL_CONFIG['chunk_size']}")
        st.text(f"• Retrieval K: {MODEL_CONFIG['retrieval_k']}")
    
    return uploaded_file, dev_mode

def render_main_interface():
    """Render the main application interface"""
    # Header
    st.title("🚀 DevAssist AI")
    st.markdown("### Developer Knowledge Assistant")
    st.markdown("---")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.qa_chain:
            st.success("✅ Document Loaded")
        else:
            st.warning("⏳ No Document")
    
    with col2:
        chat_count = len(st.session_state.chat_history)
        st.info(f"💬 {chat_count} Conversations")
    
    with col3:
        if st.session_state.current_file:
            st.info(f"📄 {st.session_state.current_file}")

def render_empty_state():
    """Render empty state when no document is uploaded"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h3>👋 Welcome to DevAssist AI!</h3>
        <p>Upload a developer document to get started.</p>
        <p><strong>Supported formats:</strong> PDF</p>
        <p><strong>Document types:</strong> PRDs, Technical Specs, API Documentation, Assignment Briefs</p>
    </div>
    """, unsafe_allow_html=True)

def render_chat_history(dev_mode: bool):
    """Render the chat history with enhanced formatting"""
    for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(question)
        
        # Assistant message
        with st.chat_message("assistant"):
            # Stream the answer
            placeholder = st.empty()
            stream_text(answer, placeholder)
            
            # Show sources
            if sources:
                st.markdown("**📚 Sources:**")
                shown_sources = set()
                
                for doc in sources:
                    source = doc.metadata.get("source", "Document")
                    page = doc.metadata.get("page", "Unknown")
                    source_key = f"{source}-{page}"
                    
                    if source_key not in shown_sources:
                        shown_sources.add(source_key)
                        st.markdown(f"• **{source}** (Page {page})")
                
                # Developer mode: show retrieved chunks
                if dev_mode:
                    with st.expander("🧠 Retrieved Content"):
                        for j, doc in enumerate(sources):
                            st.markdown(f"**Chunk {j+1}:**")
                            st.text(doc.page_content[:300] + "...")
                            st.markdown("---")

def render_developer_debug(sources, dev_mode):
    """Render developer debugging information"""
    if dev_mode and sources:
        with st.sidebar.expander("🔍 Debug Information"):
            st.metric("Chunks Retrieved", len(sources))
            st.text("Model Configuration:")
            for key, value in MODEL_CONFIG.items():
                st.text(f"• {key}: {value}")
            
            # Show retrieval scores if available
            if hasattr(sources[0], 'score'):
                st.text("Similarity Scores:")
                for i, doc in enumerate(sources):
                    st.text(f"• Chunk {i+1}: {doc.score:.3f}")

# =========================
# Main Application Logic
# =========================

def main():
    """Main application function"""
    # Initialize
    create_directories()
    initialize_session_state()
    
    # Render UI
    uploaded_file, dev_mode = render_sidebar()
    render_main_interface()
    
    # Handle file upload
    if uploaded_file:
        # Check if new file
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.processing_complete = False
            clean_temp_files()
        
        # Process document if needed
        if st.session_state.qa_chain is None:
            st.sidebar.info("🔄 Processing document...")
            
            with st.spinner("Processing document... This may take a moment."):
                qa_chain = process_document(uploaded_file)
                
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.processing_complete = True
                    st.sidebar.success("✅ Document processed successfully!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to process document")
                    return
    
    # Show empty state if no document
    if not uploaded_file:
        render_empty_state()
        return
    
    # Handle chat input
    if st.session_state.qa_chain:
        query = st.chat_input("Ask about your documentation...", key="main_chat")
        
        if query:
            # Validate question
            if not is_dev_question(query):
                answer = "❌ This question is not related to the uploaded developer document."
                sources = []
            else:
                # Process question
                with st.spinner("🤖 Analyzing document..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"query": query})
                        answer = format_answer(result["result"].strip())
                        sources = result.get("source_documents", [])
                        
                        if not sources or not answer or "could not find" in answer.lower():
                            answer = "❌ I could not find this information in the uploaded document."
                            sources = []
                            
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        answer = "❌ An error occurred while processing your question."
                        sources = []
            
            # Add to chat history
            st.session_state.chat_history.append((query, answer, sources))
            
            # Show debug info
            render_developer_debug(sources, dev_mode)
            
            # Rerun to show new message
            st.rerun()
    
    # Render chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        render_chat_history(dev_mode)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*DevAssist AI - Powered by Qwen2.5 & Sentence Transformers*",
        help="Built with Streamlit, LangChain, and Hugging Face"
    )

# =========================
# Application Entry Point
# =========================

if __name__ == "__main__":
    main()
