# 📚 PDFPal - AI Chatbot with RAG

A lightweight, chat-based RAG (Retrieval-Augmented Generation) application that allows you to upload multiple PDF documents and have intelligent conversations about their content using local, free AI models.

## ✨ Features

- **📄 Multi-PDF Support**: Upload and process multiple PDF documents simultaneously
- **🤖 Local AI Models**: Uses free, lightweight models like TinyLlama, DialoGPT, and Phi-2
- **🔍 Smart Retrieval**: FAISS vector database with sentence-transformers embeddings
- **💬 Conversational Interface**: ChatGPT-like chat experience with conversation history
- **⚙️ Configurable**: Adjustable chunk sizes, model parameters, and retrieval settings
- **🚀 Easy Deployment**: Deploy locally or to Streamlit Community Cloud
- **📊 Advanced Analytics**: Chat statistics and conversation management

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **LLM**: HuggingFace Transformers (TinyLlama, DialoGPT, Phi-2)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **RAG Framework**: LangChain
- **PDF Processing**: PyPDF2
- **Quantization**: bitsandbytes (for memory efficiency)

## 📋 Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU optional (CUDA support for faster inference)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Pdf_chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents
- Click "Choose PDF files" in the sidebar
- Select one or more PDF documents
- Click "🔄 Process Documents" to create the knowledge base

### 2. Configure Settings
- **Model Selection**: Choose from TinyLlama, DialoGPT, or Phi-2
- **Chunk Size**: Adjust text chunk size (500-2000 characters)
- **Chunk Overlap**: Set overlap between chunks (50-500 characters)
- **Max Tokens**: Control response length (100-1000 tokens)
- **Temperature**: Adjust creativity (0.0-1.0)

### 3. Start Chatting
- Type your questions in the chat input
- Ask about specific content, concepts, or request summaries
- View conversation history and clear chat when needed

## 🏗️ Architecture

The application is built with a modular architecture:

```
Pdf_chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── modules/              # Core modules
    ├── __init__.py
    ├── pdf_processor.py   # PDF text extraction & chunking
    ├── embedding_manager.py # Vector embeddings & FAISS
    ├── llm_manager.py     # Local language models
    ├── rag_pipeline.py    # RAG orchestration
    └── chat_manager.py    # Conversation management
```

### Module Descriptions

- **PDFProcessor**: Handles PDF text extraction and intelligent chunking
- **EmbeddingManager**: Manages sentence embeddings and FAISS vector database
- **LLMManager**: Loads and manages local language models with quantization
- **RAGPipeline**: Orchestrates retrieval and generation process
- **ChatManager**: Handles conversation history and chat state

## 🔧 Configuration

### Model Options

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (Recommended)
   - Size: ~1.1B parameters
   - Fast inference, good performance
   - Works well on CPU

2. **microsoft/DialoGPT-medium**
   - Size: ~345M parameters
   - Conversational model
   - Good for chat interactions

3. **microsoft/phi-2**
   - Size: ~2.7B parameters
   - High quality responses
   - Requires more memory

### Performance Tuning

- **For CPU-only systems**: Use TinyLlama or DialoGPT
- **For GPU systems**: Enable CUDA for faster inference
- **Memory optimization**: Adjust chunk size and model selection
- **Response quality**: Increase max_tokens and adjust temperature

## 🌐 Deployment

### Local Deployment

1. **Development Mode**:
   ```bash
   streamlit run app.py --server.port 8501
   ```

2. **Production Mode**:
   ```bash
   streamlit run app.py --server.headless true --server.port 8501
   ```

### Streamlit Community Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Deploy automatically

### Hugging Face Spaces

1. **Create Space**:
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Create new Space with Streamlit SDK

2. **Upload Files**:
   - Upload all project files
   - Set requirements.txt
   - Deploy automatically

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce chunk size
   - Use smaller model (TinyLlama)
   - Close other applications

2. **Model Loading Issues**:
   - Check internet connection
   - Clear HuggingFace cache: `rm -rf ~/.cache/huggingface`
   - Try different model

3. **PDF Processing Errors**:
   - Ensure PDF is not password-protected
   - Check PDF contains extractable text
   - Try different PDF

4. **Slow Performance**:
   - Enable GPU if available
   - Reduce chunk size
   - Use smaller model

### Performance Tips

- **First Run**: Models download automatically (may take time)
- **Subsequent Runs**: Models load from cache (faster)
- **GPU Usage**: Automatically detected and used if available
- **Memory Management**: Models use quantization for efficiency

## 📊 Advanced Features

### Conversation Management

- **Export Conversations**: Save chat history as JSON or text
- **Search Messages**: Find specific content in conversation history
- **Statistics**: View chat analytics and usage metrics

### Knowledge Base Management

- **Persistent Storage**: Save and load knowledge bases
- **Incremental Updates**: Add new documents to existing knowledge base
- **Similarity Search**: Find related documents with scores

### Customization

- **Custom Prompts**: Modify RAG pipeline prompts
- **Model Parameters**: Adjust generation parameters
- **Embedding Models**: Change sentence transformer models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG framework
- [HuggingFace](https://huggingface.co/) for models and transformers
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [sentence-transformers](https://www.sbert.net/) for embeddings

## 📞 Support

- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check this README and code comments

---

**Happy Chatting! 🎉**
