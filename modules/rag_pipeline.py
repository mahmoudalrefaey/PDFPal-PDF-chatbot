"""
RAG Pipeline Module
Orchestrates the retrieval-augmented generation process
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from .embedding_manager import EmbeddingManager
from .llm_manager import LLMManager

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline"""
    
    def __init__(self, knowledge_base: FAISS, llm_manager: LLMManager):
        """
        Initialize RAG pipeline
        
        Args:
            knowledge_base: FAISS vector store
            llm_manager: LLM manager instance
        """
        self.knowledge_base = knowledge_base
        self.llm_manager = llm_manager
        self.retrieval_chain = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize retrieval chain
        self._initialize_retrieval_chain()
    
    def _initialize_retrieval_chain(self):
        """Initialize the retrieval QA chain"""
        try:
            self.logger.info("Initializing retrieval QA chain")
            
            # Create custom prompt template
            prompt_template = """You are a helpful AI assistant that answers questions based on the provided context.

Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, say so. Be accurate and helpful.

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval QA chain
            self.retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm_manager.llm,
                chain_type="stuff",
                retriever=self.knowledge_base.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            self.logger.info("Retrieval QA chain initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing retrieval chain: {e}")
            raise
    
    def get_response(self, query: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Get response using RAG pipeline
        
        Args:
            query: User query
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            if not self.retrieval_chain:
                raise ValueError("Retrieval chain not initialized")
            
            self.logger.info(f"Processing query: {query[:50]}...")
            
            # Get relevant documents
            relevant_docs = self.knowledge_base.similarity_search(query, k=4)
            
            if not relevant_docs:
                return "I couldn't find any relevant information in the provided documents to answer your question."
            
            # Create context from relevant documents
            context = self._create_context(relevant_docs)
            
            # Generate response using LLM
            response = self.llm_manager.generate_response(
                prompt=self._create_prompt(query, context),
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.logger.info(f"Generated response: {response[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _create_context(self, documents: List[Document]) -> str:
        """
        Create context string from relevant documents
        
        Args:
            documents: List of relevant documents
            
        Returns:
            Context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Add document source if available
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            
            context_parts.append(f"Document {i} (Source: {source}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        return f"""Based on the following context, please answer the user's question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    def get_similar_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Get similar documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        try:
            return self.knowledge_base.similarity_search(query, k=k)
        except Exception as e:
            self.logger.error(f"Error retrieving similar documents: {e}")
            return []
    
    def get_similar_documents_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """
        Get similar documents with similarity scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        try:
            return self.knowledge_base.similarity_search_with_score(query, k=k)
        except Exception as e:
            self.logger.error(f"Error retrieving similar documents with scores: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to the knowledge base
        
        Args:
            documents: List of documents to add
        """
        try:
            if not documents:
                return
            
            self.logger.info(f"Adding {len(documents)} documents to knowledge base")
            
            # Add documents to vector store
            self.knowledge_base.add_documents(documents)
            
            # Reinitialize retrieval chain with updated knowledge base
            self._initialize_retrieval_chain()
            
            self.logger.info("Documents added successfully")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG pipeline
        
        Returns:
            Dictionary with pipeline information
        """
        try:
            # Get knowledge base info
            kb_info = {}
            if self.knowledge_base:
                index = self.knowledge_base.index
                kb_info = {
                    "documents": index.ntotal if hasattr(index, 'ntotal') else "unknown",
                    "index_type": type(index).__name__
                }
            
            # Get LLM info
            llm_info = self.llm_manager.get_model_info()
            
            return {
                "status": "initialized" if self.retrieval_chain else "not_initialized",
                "knowledge_base": kb_info,
                "language_model": llm_info,
                "retrieval_chain": "initialized" if self.retrieval_chain else "not_initialized"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline info: {e}")
            return {"status": "error", "error": str(e)}
    
    def update_retrieval_parameters(self, k: int = 4, search_type: str = "similarity"):
        """
        Update retrieval parameters
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search (similarity, mmr, etc.)
        """
        try:
            self.logger.info(f"Updating retrieval parameters: k={k}, search_type={search_type}")
            
            # Update retriever
            self.knowledge_base.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )
            
            # Reinitialize chain
            self._initialize_retrieval_chain()
            
            self.logger.info("Retrieval parameters updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating retrieval parameters: {e}")
            raise 