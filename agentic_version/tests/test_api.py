#!/usr/bin/env python3
"""
OpenAI API tests for the agentic AI system
"""

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_openai_api():
    """Test OpenAI API connection"""
    print("🧪 Testing OpenAI API connection...")
    
    try:
        from config.config import Config
        import openai
        
        print(f"API Key (first 10 chars): {Config.OPENAI_API_KEY[:10]}...")
        print(f"Model: {Config.OPENAI_MODEL}")
        
        # Create client
        client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Test simple completion
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "Say 'Hello, API test successful!'"}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API Test Successful!")
        print(f"Response: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        return False

def test_llm_manager():
    """Test LLM manager functionality"""
    print("\n🧪 Testing LLM Manager...")
    
    try:
        from tools.llm_manager import LLMManager
        
        # Create LLM manager
        llm_manager = LLMManager()
        print("✅ LLM Manager created successfully")
        
        # Test SQL generation
        test_schemas = {
            'test_table': {
                'columns': [
                    {'name': 'id', 'type': 'integer', 'nullable': False},
                    {'name': 'name', 'type': 'text', 'nullable': True},
                    {'name': 'value', 'type': 'numeric', 'nullable': True}
                ],
                'sample_data': [{'id': 1, 'name': 'test', 'value': 100}]
            }
        }
        
        result = llm_manager.generate_sql_query(
            "Show me all records from test_table", 
            test_schemas
        )
        
        if result['success']:
            print(f"✅ SQL Generation successful!")
            print(f"Generated SQL: {result['sql_query']}")
        else:
            print(f"❌ SQL Generation failed: {result['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ LLM Manager test failed: {e}")
        return False

def run_api_tests():
    """Run all API tests"""
    print("🚀 OpenAI API Test")
    print("=" * 30)
    
    # Test API connection
    api_success = test_openai_api()
    
    if api_success:
        # Test LLM manager
        llm_success = test_llm_manager()
        
        if llm_success:
            print("\n🎉 All API tests passed! OpenAI API is working correctly.")
            return True
        else:
            print("\n⚠️ LLM Manager test failed.")
            return False
    else:
        print("\n❌ API connection failed. Please check your API key.")
        return False

if __name__ == "__main__":
    success = run_api_tests()
    sys.exit(0 if success else 1)
