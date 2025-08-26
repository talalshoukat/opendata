#!/usr/bin/env python3
"""
Basic functionality tests for the agentic AI system
"""

import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_imports():
    """Test that core modules can be imported"""
    print("🧪 Testing basic imports...")
    
    try:
        from config.state import AgentState, ToolResult
        print("✅ State management imported")
        
        from config.config import Config
        print("✅ Configuration imported")
        
        from tools.vector_store import FAISSVectorStore
        print("✅ Vector store imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_state_management():
    """Test state management functionality"""
    print("\n🧪 Testing state management...")
    
    try:
        from config.state import AgentState, ToolResult
        
        # Test state creation
        state = AgentState(user_query="test query")
        assert state.user_query == "test query"
        assert state.current_step == "start"
        print("✅ State creation works")
        
        # Test state updates
        state.update_step("new_step")
        assert state.current_step == "new_step"
        print("✅ State updates work")
        
        # Test tool call tracking
        mock_result = ToolResult(success=True, data="test_data")
        state.add_tool_call("test_tool", "test_input", mock_result)
        assert len(state.tool_calls) == 1
        print("✅ Tool call tracking works")
        
        # Test error handling
        state.add_error("test error")
        assert len(state.errors) == 1
        print("✅ Error handling works")
        
        return True
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\n🧪 Testing vector store...")
    
    try:
        from tools.vector_store import FAISSVectorStore
        
        # Create vector store
        vector_store = FAISSVectorStore()
        print("✅ Vector store created")
        
        # Test adding categorical values
        test_data = {
            'occupation': ['Engineer', 'Manager', 'Analyst'],
            'sector': ['Technology', 'Finance', 'Healthcare']
        }
        vector_store.add_categorical_values('test_table', test_data)
        print("✅ Categorical values added")
        
        # Test building index
        vector_store.build_index()
        print("✅ Index built")
        
        # Test search functionality
        results = vector_store.search_similar_keywords('engineer')
        print(f"✅ Search works: found {len(results)} results")
        
        # Test normalization
        normalized = vector_store.normalize_query_keywords("show me engineers in tech")
        print(f"✅ Normalization works: {normalized['success']}")
        
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_tool_definitions():
    """Test tool definitions with mocked dependencies"""
    print("\n🧪 Testing tool definitions...")
    
    try:
        from tools.tool_definitions import create_tools
        
        # Mock dependencies
        mock_db = Mock()
        mock_vector = Mock()
        mock_llm = Mock()
        
        # Create tools
        tools = create_tools(mock_db, mock_vector, mock_llm)
        
        # Verify tools were created
        assert len(tools) == 5
        tool_names = [tool.name for tool in tools]
        expected_names = ['keyword_normalizer', 'schema_inspector', 'sql_generator', 'db_query', 'result_formatter']
        
        for name in expected_names:
            assert name in tool_names
        
        print(f"✅ Tools created: {tool_names}")
        return True
        
    except Exception as e:
        print(f"❌ Tool definitions test failed: {e}")
        return False

def test_workflow_logic():
    """Test basic workflow logic without external dependencies"""
    print("\n🧪 Testing workflow logic...")
    
    try:
        from config.state import AgentState
        
        # Create a test state
        state = AgentState(user_query="test query")
        
        # Simulate workflow steps
        steps = ["start", "normalize_keywords", "inspect_schema", "generate_sql", "execute_query", "format_results"]
        
        for step in steps:
            state.update_step(step)
            state.add_tool_call(f"test_{step}", "test_input", 
                              Mock(success=True, data="test_data"))
        
        # Verify workflow progression
        assert state.current_step == "format_results"
        assert len(state.tool_calls) == len(steps)
        
        # Set workflow as complete
        state.should_continue = False
        state.final_response = "Test response"
        state.sql_execution_result = "Test results"
        assert state.is_complete()
        
        print("✅ Workflow logic works")
        return True
        
    except Exception as e:
        print(f"❌ Workflow logic test failed: {e}")
        return False

def run_basic_tests():
    """Run all basic tests"""
    print("🚀 Running Basic Functionality Tests\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("State Management", test_state_management),
        ("Vector Store", test_vector_store),
        ("Tool Definitions", test_tool_definitions),
        ("Workflow Logic", test_workflow_logic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The core system is working.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
