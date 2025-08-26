#!/usr/bin/env python3
"""
Setup script for environment configuration
"""

import os
import sys

def create_env_file():
    """Create a .env file with default values"""
    env_content = """# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=open_data
DB_USER=talal
DB_PASSWORD=my_password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Vector Store Configuration
VECTOR_STORE_PATH=./vector_store

# Agent Configuration
MAX_RETRIES=3
MAX_TOOL_CALLS=10

# LLM Configuration
TEMPERATURE=0.1
MAX_TOKENS=4000

# Vector Search Configuration
SIMILARITY_THRESHOLD=0.7
MAX_CANDIDATES=10
"""
    
    env_file = ".env"
    
    if os.path.exists(env_file):
        print(f"⚠️  {env_file} already exists. Skipping creation.")
        return False
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created {env_file} with default values")
        print("📝 Please edit the file to set your actual configuration values")
        return True
    except Exception as e:
        print(f"❌ Error creating {env_file}: {e}")
        return False

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
        return False
    
    # Check required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'DB_HOST',
        'DB_NAME',
        'DB_USER',
        'DB_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

def main():
    """Main setup function"""
    print("🎯 Environment Setup for Agentic AI System")
    print("=" * 50)
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("📝 Creating .env file...")
        create_env_file()
        print("\n📋 Next steps:")
        print("1. Edit the .env file with your actual configuration values")
        print("2. Set your OpenAI API key")
        print("3. Update database connection details if needed")
        print("4. Run this script again to verify configuration")
    else:
        # Check environment
        if check_environment():
            print("\n🎉 Environment is properly configured!")
            print("✅ You can now run the system")
        else:
            print("\n⚠️  Environment needs configuration")
            print("📝 Please edit the .env file and set the required values")

if __name__ == "__main__":
    main()
