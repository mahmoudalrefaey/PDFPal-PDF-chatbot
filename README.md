# 📚 PDFPal - AI Chatbot with RAG

A lightweight, chat-based Retrieval-Augmented Generation (RAG) application that lets you upload multiple PDF documents and interact with them using local, free AI models—all through a modern Streamlit interface.

---

## ✨ Features
- **Multi-PDF Support:** Upload and process multiple PDF files at once
- **Local AI Models:** Use free, open-source models like TinyLlama, DialoGPT, and Phi-2
- **Smart Retrieval:** FAISS vector database with sentence-transformers embeddings
- **Conversational UI:** ChatGPT-like experience with persistent chat history
- **Configurable:** Adjust chunk size, overlap, model, and generation parameters
- **Easy Deployment:** Run locally or deploy to Streamlit Community Cloud

---

## 🛠️ Technology Stack
- **Frontend:** Streamlit
- **LLM:** HuggingFace Transformers (TinyLlama, DialoGPT, Phi-2)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** FAISS
- **RAG Framework:** LangChain
- **PDF Processing:** PyPDF2
- **Quantization:** bitsandbytes (for memory efficiency)

---

## 📋 Requirements
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU optional (CUDA for faster inference)

---

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd Pdf_chatbot
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

1. **Upload Documents**
   - Use the sidebar to upload one or more PDF files
   - Click "Process Documents" to build the knowledge base
2. **Configure Settings**
   - Select the LLM model (TinyLlama, DialoGPT, or Phi-2)
   - Adjust chunk size, overlap, max tokens, and temperature as needed
3. **Chat**
   - Enter your questions in the chat input
   - View the conversation history and clear it when needed

---

## 🏗️ Project Structure

```
Pdf_chatbot/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── config.py              # Centralized configuration
├── test_installation.py   # Environment/test script
└── modules/               # Core modules
    ├── __init__.py
    ├── pdf_processor.py       # PDF text extraction & chunking
    ├── embedding_manager.py   # Vector embeddings & FAISS
    ├── llm_manager.py         # Local language models
    ├── rag_pipeline.py        # RAG orchestration
    └── chat_manager.py        # Conversation management
```

---

## 🔧 Configuration & Models

- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (Recommended, fast, CPU-friendly)
- **microsoft/DialoGPT-medium** (Conversational, lightweight)
- **microsoft/phi-2** (Higher quality, more memory required)

**Performance Tips:**
- Use smaller models for CPU or low-memory systems
- Reduce chunk size and max tokens for faster responses
- Enable GPU for best performance (if available)

---

## 🌐 Deployment

### Local
```bash
streamlit run app.py
```

### Streamlit Community Cloud
1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy

---

## 🔍 Troubleshooting
- **Memory errors:** Lower chunk size or use a smaller model
- **Model loading issues:** Check your internet connection and HuggingFace cache
- **PDF errors:** Ensure PDFs are not password-protected and contain extractable text
- **Slow performance:** Use a smaller model, reduce chunk size, or enable GPU

---

## 📊 Advanced Features
- **Export/Save chat history** (coming soon)
- **Search messages** (coming soon)
- **Custom prompts and retrieval settings**

---

## 🤝 Contributing
1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License
MIT License

---

## 🙏 Acknowledgments
- [LangChain](https://langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)

---

**Happy Chatting! 🎉**
