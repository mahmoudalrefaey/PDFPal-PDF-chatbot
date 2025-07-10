"""
Configuration file for PDFPal RAG application
Centralized settings for easy customization
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    
    # Application settings
    APP_NAME = "PDFPal - AI Chatbot"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model configurations
    DEFAULT_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Available models
    AVAILABLE_MODELS = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "name": "TinyLlama 1.1B",
            "description": "Fast and efficient 1.1B parameter model",
            "size": "1.1B",
            "recommended": True
        },
        "microsoft/DialoGPT-medium": {
            "name": "DialoGPT Medium",
            "description": "Conversational model optimized for chat",
            "size": "345M",
            "recommended": False
        },
        "microsoft/phi-2": {
            "name": "Phi-2",
            "description": "High-quality 2.7B parameter model",
            "size": "2.7B",
            "recommended": False
        }
    }
    
    # Default processing settings
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_TEMPERATURE = 0.7
    
    # RAG settings
    DEFAULT_RETRIEVAL_K = 4
    DEFAULT_SEARCH_TYPE = "similarity"
    
    # Chat settings
    MAX_CHAT_HISTORY = 100
    MAX_CONTEXT_MESSAGES = 5
    
    # File settings
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FILE_TYPES = ["pdf"]
    
    # Performance settings
    ENABLE_GPU = True
    ENABLE_QUANTIZATION = True
    CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
    
    # UI settings
    SIDEBAR_EXPANDED = True
    PAGE_LAYOUT = "wide"
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return cls.AVAILABLE_MODELS.get(model_name, cls.AVAILABLE_MODELS[cls.DEFAULT_LLM_MODEL])
    
    @classmethod
    def get_recommended_model(cls) -> str:
        """Get the recommended model name"""
        for model_name, config in cls.AVAILABLE_MODELS.items():
            if config.get("recommended", False):
                return model_name
        return cls.DEFAULT_LLM_MODEL
    
    @classmethod
    def get_model_names(cls) -> list:
        """Get list of available model names"""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """Validate if a model name is supported"""
        return model_name in cls.AVAILABLE_MODELS
    
    @classmethod
    def get_ui_config(cls) -> Dict[str, Any]:
        """Get UI configuration"""
        return {
            "page_title": cls.APP_NAME,
            "page_icon": "📚",
            "layout": cls.PAGE_LAYOUT,
            "initial_sidebar_state": "expanded" if cls.SIDEBAR_EXPANDED else "collapsed"
        } 