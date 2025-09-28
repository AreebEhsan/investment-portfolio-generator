#!/usr/bin/env python3
"""
Simple launcher script for the Investment Portfolio Recommendation Engine.
This script can be used as an alternative to running 'streamlit run app.py' directly.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import yfinance
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has API keys."""
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  .env file not found. Copying from template...")
        template_path = Path(".env.template")
        if template_path.exists():
            import shutil
            shutil.copy(template_path, env_path)
            print("✅ .env file created. Please add your API keys to .env file.")
        else:
            print("❌ .env.template not found. Please create .env file manually.")
        return False
    
    # Check if Alpha Vantage key is configured
    with open(env_path, 'r') as f:
        content = f.read()
        if "your_free_alpha_vantage_api_key_here" in content:
            print("⚠️  Please configure your Alpha Vantage API key in .env file")
            return False
    
    return True

def main():
    """Main function to launch the Streamlit app."""
    print("🚀 Starting Investment Portfolio Recommendation Engine...")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        print("📝 Configuration needed. Please check .env file and try again.")
        input("Press Enter to continue anyway...")
    
    # Launch Streamlit app
    print("🌐 Launching Streamlit application...")
    print("📊 Your portfolio optimizer will open in your browser shortly...")
    print("\n💡 Tip: Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--theme.base", "light",
            "--theme.primaryColor", "#1f77b4"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("💡 Try running directly: streamlit run app.py")

if __name__ == "__main__":
    main()