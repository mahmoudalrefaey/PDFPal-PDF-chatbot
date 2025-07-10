"""
Embedding Manager Module
Handles text embeddings and vector database operations using sentence-transformers and FAISS
"""

import logging
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingManager:
    """Manages text embeddings and vector database operations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.embeddings = None
        self.vector_store = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            
            # Use HuggingFaceEmbeddings wrapper for better LangChain integration
            self.embeddings = HuggingFaceEmbeddings(
                model_name=f"sentence-transformers/{self.model_name}",
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise
    
    def create_knowledge_base(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS knowledge base from documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store
        """
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            self.logger.info(f"Creating knowledge base with {len(documents)} documents")
            
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            self.logger.info("Knowledge base created successfully")
            return self.vector_store
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge base: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the knowledge base
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        try:
            if not self.vector_store:
                raise ValueError("Knowledge base not initialized")
            
            self.logger.info(f"Performing similarity search for: {query[:50]}...")
            
            # Perform search
            similar_docs = self.vector_store.similarity_search(query, k=k)
            
            self.logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with scores
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if not self.vector_store:
                raise ValueError("Knowledge base not initialized")
            
            self.logger.info(f"Performing similarity search with scores for: {query[:50]}...")
            
            # Perform search with scores
            similar_docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            self.logger.info(f"Found {len(similar_docs_with_scores)} similar documents")
            return similar_docs_with_scores
            
        except Exception as e:
            self.logger.error(f"Error in similarity search with scores: {e}")
            raise
    
    def save_knowledge_base(self, path: str = "knowledge_base"):
        """
        Save the knowledge base to disk
        
        Args:
            path: Directory path to save the knowledge base
        """
        try:
            if not self.vector_store:
                raise ValueError("Knowledge base not initialized")
            
            self.logger.info(f"Saving knowledge base to: {path}")
            
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save vector store
            self.vector_store.save_local(path)
            
            self.logger.info("Knowledge base saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")
            raise
    
    def load_knowledge_base(self, path: str = "knowledge_base") -> FAISS:
        """
        Load the knowledge base from disk
        
        Args:
            path: Directory path to load the knowledge base from
            
        Returns:
            FAISS vector store
        """
        try:
            self.logger.info(f"Loading knowledge base from: {path}")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Knowledge base not found at: {path}")
            
            # Load vector store
            self.vector_store = FAISS.load_local(path, self.embeddings)
            
            self.logger.info("Knowledge base loaded successfully")
            return self.vector_store
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            raise
    
    def get_knowledge_base_info(self) -> dict:
        """
        Get information about the knowledge base
        
        Returns:
            Dictionary with knowledge base information
        """
        if not self.vector_store:
            return {"status": "not_initialized", "documents": 0}
        
        try:
            # Get index info
            index = self.vector_store.index
            num_docs = index.ntotal if hasattr(index, 'ntotal') else "unknown"
            
            return {
                "status": "initialized",
                "documents": num_docs,
                "embedding_model": self.model_name,
                "index_type": type(index).__name__
            }
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge base info: {e}")
            return {"status": "error", "error": str(e)} 