#!/usr/bin/env python3
"""
Deployment script for PDFPal RAG application
Helps deploy to different platforms easily
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_requirements() -> bool:
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found")
        return False
    
    print("✅ Requirements check passed")
    return True

def install_dependencies() -> bool:
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def run_local() -> bool:
    """Run the application locally"""
    print("🚀 Starting PDFPal locally...")
    print("📱 The application will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
        return True
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")
        return False

def setup_git() -> bool:
    """Setup Git repository"""
    if not Path(".git").exists():
        return run_command("git init", "Initializing Git repository")
    return True

def create_streamlit_config() -> bool:
    """Create Streamlit configuration file"""
    config_content = """[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
"""
    
    try:
        with open(".streamlit/config.toml", "w") as f:
            f.write(config_content)
        print("✅ Created Streamlit configuration")
        return True
    except Exception as e:
        print(f"❌ Failed to create Streamlit config: {e}")
        return False

def deploy_streamlit_cloud() -> bool:
    """Deploy to Streamlit Community Cloud"""
    print("🌐 Deploying to Streamlit Community Cloud...")
    
    # Setup Git
    if not setup_git():
        return False
    
    # Create .streamlit directory and config
    os.makedirs(".streamlit", exist_ok=True)
    if not create_streamlit_config():
        return False
    
    # Add files to Git
    commands = [
        "git add .",
        'git commit -m "Initial deployment"',
        "git branch -M main"
    ]
    
    for command in commands:
        if not run_command(command, f"Running: {command}"):
            return False
    
    print("✅ Ready for Streamlit Cloud deployment!")
    print("📋 Next steps:")
    print("1. Push to GitHub: git remote add origin <your-repo-url>")
    print("2. Push: git push -u origin main")
    print("3. Go to https://share.streamlit.io")
    print("4. Connect your GitHub repository")
    print("5. Deploy!")
    
    return True

def deploy_huggingface() -> bool:
    """Deploy to Hugging Face Spaces"""
    print("🤗 Deploying to Hugging Face Spaces...")
    
    # Create app.py for HF Spaces
    hf_app_content = '''import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("hf_app.py", "w") as f:
            f.write(hf_app_content)
        print("✅ Created Hugging Face app file")
    except Exception as e:
        print(f"❌ Failed to create HF app file: {e}")
        return False
    
    print("✅ Ready for Hugging Face Spaces deployment!")
    print("📋 Next steps:")
    print("1. Go to https://huggingface.co/spaces")
    print("2. Create new Space with Streamlit SDK")
    print("3. Upload all project files")
    print("4. Set requirements.txt")
    print("5. Deploy!")
    
    return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy PDFPal RAG application")
    parser.add_argument("--mode", choices=["local", "streamlit", "huggingface", "check"], 
                       default="local", help="Deployment mode")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    
    args = parser.parse_args()
    
    print("📚 PDFPal Deployment Script")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install:
        if not install_dependencies():
            sys.exit(1)
    
    # Run based on mode
    if args.mode == "check":
        print("✅ All checks passed!")
        return
    
    elif args.mode == "local":
        if not install_dependencies():
            sys.exit(1)
        if not run_local():
            sys.exit(1)
    
    elif args.mode == "streamlit":
        if not deploy_streamlit_cloud():
            sys.exit(1)
    
    elif args.mode == "huggingface":
        if not deploy_huggingface():
            sys.exit(1)
    
    print("🎉 Deployment completed successfully!")

if __name__ == "__main__":
    main() 