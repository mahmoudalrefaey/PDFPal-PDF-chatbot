"""
PDFPal - A lightweight, chat-based RAG application
Built with free, local models and deployable via Streamlit
"""

import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from pathlib import Path

# Import our custom modules
from modules.pdf_processor import PDFProcessor
from modules.embedding_manager import EmbeddingManager
from modules.llm_manager import LLMManager
from modules.rag_pipeline import RAGPipeline
from modules.chat_manager import ChatManager

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="PDFPal - AI Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("📚 PDFPal")
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Configuration")
        llm_model = st.selectbox(
            "Choose LLM Model:",
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft/DialoGPT-medium", "microsoft/phi-2"],
            help="Select a lightweight local model"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
            max_tokens = st.slider("Max Response Tokens", 100, 1000, 500, 50)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        st.markdown("---")
        
        # File upload section
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload multiple PDF files for context"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) uploaded")
            
            # Process files button
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    process_uploaded_files(uploaded_files, chunk_size, chunk_overlap, llm_model)
    
    # Main chat interface
    st.title("💬 PDFPal Chat")
    st.markdown("Ask questions about your uploaded PDF documents!")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input and st.session_state.rag_pipeline:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.rag_pipeline.get_response(
                user_input, 
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        # Add AI response to chat
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update chat display
        st.rerun()
    
    elif user_input and not st.session_state.rag_pipeline:
        st.error("⚠️ Please upload and process documents first!")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def process_uploaded_files(uploaded_files: List, chunk_size: int, chunk_overlap: int, llm_model: str):
    """Process uploaded PDF files and create knowledge base"""
    try:
        # Initialize components
        pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedding_manager = EmbeddingManager()
        llm_manager = LLMManager(model_name=llm_model)
        
        # Process all files
        all_chunks = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process PDF
                chunks = pdf_processor.process_pdf(tmp_path)
                all_chunks.extend(chunks)
                st.session_state.uploaded_files.append(uploaded_file.name)
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        
        if all_chunks:
            # Create knowledge base
            knowledge_base = embedding_manager.create_knowledge_base(all_chunks)
            st.session_state.knowledge_base = knowledge_base
            
            # Initialize RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(
                knowledge_base=knowledge_base,
                llm_manager=llm_manager
            )
            
            st.success(f"✅ Processed {len(all_chunks)} text chunks from {len(uploaded_files)} file(s)")
        else:
            st.error("❌ No text could be extracted from the uploaded files")
            
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")

def display_chat_history():
    """Display the chat history in a conversational format"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

if __name__ == "__main__":
    main()

