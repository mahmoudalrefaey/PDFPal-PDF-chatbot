"""
PDF Processor Module
Handles PDF text extraction and chunking for RAG pipeline
"""

import logging
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class PDFProcessor:
    """Handles PDF processing, text extraction, and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            self.logger.info(f"Extracting text from: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
                
                self.logger.info(f"Extracted {len(text)} characters from PDF")
                return text
                
        except Exception as e:
            self.logger.error(f"Error reading PDF file {pdf_path}: {e}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[Document]:
        """
        Split text into chunks using LangChain text splitter
        
        Args:
            text: Text to split
            
        Returns:
            List of Document objects
        """
        try:
            self.logger.info("Splitting text into chunks")
            
            # Create a single document first
            documents = [Document(page_content=text, metadata={"source": "pdf"})]
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            self.logger.info(f"Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error splitting text: {e}")
            raise
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document chunks
        """
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                self.logger.warning("No text extracted from PDF")
                return []
            
            # Split into chunks
            chunks = self.split_text_into_chunks(text)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata["source"] = pdf_path
                chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunks
        
        Args:
            chunks: List of Document chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0, "avg_chunk_size": 0, "total_characters": 0}
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_size = total_chars / len(chunks)
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(avg_size, 2),
            "total_characters": total_chars
        } 