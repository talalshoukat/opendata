#!/usr/bin/env python3
"""
Simple launcher for the FastAPI Chat Interface
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching FastAPI Chat Interface...")
    print("=" * 50)
    print("💬 Chat interface will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chat Interface stopped by user")

if __name__ == "__main__":
    main()


