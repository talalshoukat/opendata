#!/usr/bin/env python3
"""
Simple launcher for the Streamlit frontend
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching Agentic AI Frontend...")
    print("=" * 50)
    print("📱 Opening browser to: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")

if __name__ == "__main__":
    main()
