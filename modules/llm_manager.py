"""
LLM Manager Module
Handles local language models using transformers and HuggingFace
"""

import logging
import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from langchain_community.llms import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManager

class LLMManager:
    """Manages local language models for text generation"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize LLM manager
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.llm = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_config = {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "max_length": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "pad_token_id": 0,
                "eos_token_id": 2
            },
            "microsoft/DialoGPT-medium": {
                "max_length": 1000,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": 50256,
                "eos_token_id": 50256
            },
            "microsoft/phi-2": {
                "max_length": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "pad_token_id": 0,
                "eos_token_id": 50256
            }
        }
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the language model"""
        try:
            self.logger.info(f"Loading language model: {self.model_name}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization for memory efficiency
            if device == "cuda":
                # Use 4-bit quantization for GPU
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                )
            else:
                # Use CPU with 8-bit quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            # Get model configuration
            config = self.model_config.get(self.model_name, self.model_config["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=config["max_length"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                do_sample=config["do_sample"],
                pad_token_id=config["pad_token_id"],
                eos_token_id=config["eos_token_id"],
                return_full_text=False
            )
            
            # Create LangChain LLM wrapper
            self.llm = HuggingFacePipeline(
                pipeline=self.pipeline,
                model_kwargs={"temperature": config["temperature"]}
            )
            
            self.logger.info("Language model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading language model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate response using the language model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            if not self.llm:
                raise ValueError("Language model not initialized")
            
            self.logger.info(f"Generating response for prompt: {prompt[:50]}...")
            
            # Format prompt based on model
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate response
            response = self.llm(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            # Clean up response
            cleaned_response = self._clean_response(response)
            
            self.logger.info(f"Generated response: {cleaned_response[:50]}...")
            return cleaned_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt based on the model type
        
        Args:
            prompt: Raw prompt
            
        Returns:
            Formatted prompt
        """
        if "TinyLlama" in self.model_name:
            # TinyLlama chat format
            return f"<|system|>You are a helpful AI assistant. Answer questions based on the provided context.</s><|user|>{prompt}</s><|assistant|>"
        elif "DialoGPT" in self.model_name:
            # DialoGPT format
            return f"User: {prompt}\nAssistant:"
        elif "phi" in self.model_name:
            # Phi format
            return f"Instruct: {prompt}\nOutput:"
        else:
            # Default format
            return prompt
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up the generated response
        
        Args:
            response: Raw response
            
        Returns:
            Cleaned response
        """
        # Remove prompt from response if present
        if "Instruct:" in response:
            response = response.split("Output:")[-1].strip()
        elif "User:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Remove any remaining special tokens
        response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"status": "not_initialized"}
        
        try:
            # Get model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "status": "initialized",
                "model_name": self.model_name,
                "total_parameters": f"{total_params:,}",
                "trainable_parameters": f"{trainable_params:,}",
                "device": next(self.model.parameters()).device,
                "dtype": str(next(self.model.parameters()).dtype)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {"status": "error", "error": str(e)}
    
    def change_model(self, model_name: str):
        """
        Change the language model
        
        Args:
            model_name: New model name
        """
        try:
            self.logger.info(f"Changing model from {self.model_name} to {model_name}")
            
            # Update model name
            self.model_name = model_name
            
            # Clear existing model
            self.tokenizer = None
            self.model = None
            self.pipeline = None
            self.llm = None
            
            # Reinitialize with new model
            self._initialize_model()
            
            self.logger.info("Model changed successfully")
            
        except Exception as e:
            self.logger.error(f"Error changing model: {e}")
            raise 