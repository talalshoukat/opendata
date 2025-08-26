#!/usr/bin/env python3
"""
Launcher script for the Streamlit frontend
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit frontend"""
    print("🚀 Launching Agentic AI Frontend...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run this script from the frontend directory.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("\n🌐 Starting Streamlit server...")
    print("📱 The application will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("\n" + "=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error launching frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
