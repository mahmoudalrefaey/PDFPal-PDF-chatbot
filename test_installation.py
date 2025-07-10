#!/usr/bin/env python3
"""
Test script for PDFPal RAG application
Verifies that all components are working correctly
"""

import sys
import os
import importlib
from pathlib import Path

def test_import(module_name: str, description: str) -> bool:
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False

def test_file_exists(file_path: str, description: str) -> bool:
    """Test if a file exists"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - File not found")
        return False

def test_python_version() -> bool:
    """Test Python version"""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (3.8+ required)")
        return False

def test_modules() -> bool:
    """Test all custom modules"""
    print("\n🔍 Testing custom modules...")
    
    modules = [
        ("modules", "Modules package"),
        ("modules.pdf_processor", "PDF Processor"),
        ("modules.embedding_manager", "Embedding Manager"),
        ("modules.llm_manager", "LLM Manager"),
        ("modules.rag_pipeline", "RAG Pipeline"),
        ("modules.chat_manager", "Chat Manager"),
    ]
    
    success = True
    for module_name, description in modules:
        if not test_import(module_name, description):
            success = False
    
    return success

def test_dependencies() -> bool:
    """Test external dependencies"""
    print("\n🔍 Testing external dependencies...")
    
    dependencies = [
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("PyPDF2", "PyPDF2"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
    ]
    
    success = True
    for module_name, description in dependencies:
        if not test_import(module_name, description):
            success = False
    
    return success

def test_files() -> bool:
    """Test required files"""
    print("\n🔍 Testing required files...")
    
    files = [
        ("app.py", "Main application"),
        ("requirements.txt", "Requirements file"),
        ("README.md", "Documentation"),
        ("config.py", "Configuration"),
        ("deploy.py", "Deployment script"),
        ("modules/__init__.py", "Modules init"),
        ("modules/pdf_processor.py", "PDF Processor module"),
        ("modules/embedding_manager.py", "Embedding Manager module"),
        ("modules/llm_manager.py", "LLM Manager module"),
        ("modules/rag_pipeline.py", "RAG Pipeline module"),
        ("modules/chat_manager.py", "Chat Manager module"),
    ]
    
    success = True
    for file_path, description in files:
        if not test_file_exists(file_path, description):
            success = False
    
    return success

def test_gpu_availability() -> bool:
    """Test GPU availability"""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("ℹ️  GPU not available - will use CPU")
            return True
    except ImportError:
        print("❌ PyTorch not available")
        return False

def test_model_download() -> bool:
    """Test model download capability"""
    print("\n🔍 Testing model download capability...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try to load a small model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load sentence transformer model: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 PDFPal Installation Test")
    print("=" * 40)
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Files", test_files),
        ("External Dependencies", test_dependencies),
        ("Custom Modules", test_modules),
        ("GPU Availability", test_gpu_availability),
        ("Model Download", test_model_download),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PDFPal is ready to use.")
        print("\n🚀 To start the application:")
        print("   streamlit run app.py")
        print("\n📖 For more information, see README.md")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Common solutions:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check Python version (3.8+ required)")
        print("   3. Ensure all files are present")
        print("   4. Check internet connection for model downloads")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 