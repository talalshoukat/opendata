#!/usr/bin/env python3
"""
Example usage script for the Agentic AI System

This script demonstrates how to use the system programmatically,
including error handling and custom configurations.
"""

import os
import sys
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """Example 1: Basic usage of the agentic system"""
    print("🔧 Example 1: Basic Usage")
    print("=" * 50)
    
    try:
        from agents.agent_planner import AgentPlanner
        
        # Initialize the agent
        print("🚀 Initializing Agent Planner...")
        agent = AgentPlanner()
        
        # Process a simple query
        query = "Show me the distribution of contributors by economic activity"
        print(f"📝 Processing query: {query}")
        
        result = agent.process_query(query)
        
        # Check if successful
        if result.is_complete():
            print("✅ Query processed successfully!")
            print(f"💻 SQL Generated: {result.generated_sql}")
            print(f"💬 Response: {result.final_response[:100]}...")
            if result.visualization_code:
                print(f"📈 Visualization code generated ({len(result.visualization_code)} chars)")
        else:
            print("⚠️  Query processing encountered issues")
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"  - {error}")
        
        # Clean up
        agent.close()
        
    except Exception as e:
        print(f"❌ Error in basic usage example: {e}")

def example_custom_tools():
    """Example 2: Using individual tools directly"""
    print("\n🔧 Example 2: Custom Tool Usage")
    print("=" * 50)
    
    try:
        from tools.database_manager import DatabaseManager
        from tools.vector_store import FAISSVectorStore
        from tools.llm_manager import LLMManager
        from tools.tool_definitions import create_tools
        
        # Initialize individual components
        print("🚀 Initializing individual tools...")
        db_manager = DatabaseManager()
        vector_store = FAISSVectorStore()
        llm_manager = LLMManager()
        
        # Create tools
        tools = create_tools(db_manager, vector_store, llm_manager)
        
        # Use specific tools
        print("🔍 Using schema inspector tool...")
        schema_tool = tools[1]  # SchemaInspectorTool
        schema_result = schema_tool._run()
        
        if schema_result.success:
            print(f"✅ Schema inspection successful: {len(schema_result.data)} tables")
        else:
            print(f"❌ Schema inspection failed: {schema_result.error}")
        
        # Clean up
        db_manager.close()
        
    except Exception as e:
        print(f"❌ Error in custom tools example: {e}")

def example_error_handling():
    """Example 3: Error handling and retry logic"""
    print("\n🔧 Example 3: Error Handling")
    print("=" * 50)
    
    try:
        from config.state import AgentState
        
        # Create a state with errors
        state = AgentState(user_query="test query")
        
        # Simulate errors
        state.add_error("Database connection failed")
        state.add_error("SQL generation failed")
        
        print(f"📊 Current state: {state.current_step}")
        print(f"❌ Errors: {len(state.errors)}")
        print(f"🔄 Retry count: {state.retry_count}")
        
        # Test retry logic
        state.increment_retry()
        print(f"🔄 After retry: {state.retry_count}")
        
        if state.retry_count >= state.max_retries:
            state.should_continue = False
            print("🛑 Maximum retries reached, stopping workflow")
        
        print(f"✅ Should continue: {state.should_continue}")
        
    except Exception as e:
        print(f"❌ Error in error handling example: {e}")

def example_workflow_monitoring():
    """Example 4: Monitoring workflow execution"""
    print("\n🔧 Example 4: Workflow Monitoring")
    print("=" * 50)
    
    try:
        from config.state import AgentState, ToolResult
        
        # Create a state and simulate tool executions
        state = AgentState(user_query="monitoring test query")
        
        # Simulate tool executions
        tool_results = [
            ("keyword_normalizer", "start", True, "Keywords normalized successfully"),
            ("schema_inspector", "inspect_schema", True, "Schema retrieved"),
            ("sql_generator", "generate_sql", False, "SQL generation failed"),
        ]
        
        for tool_name, step, success, message in tool_results:
            result = ToolResult(
                success=success,
                data=message,
                error=None if success else "Tool execution failed"
            )
            state.add_tool_call(tool_name, f"input for {tool_name}", result)
            state.update_step(step)
        
        # Monitor the workflow
        print(f"📊 Workflow Status: {state.current_step}")
        print(f"🔧 Tools Executed: {len(state.tool_calls)}")
        
        # Show tool execution history
        print("\n📋 Tool Execution History:")
        for i, tool_call in enumerate(state.tool_calls, 1):
            status = "✅" if tool_call['result'].success else "❌"
            print(f"  {i}. {status} {tool_call['tool']} ({tool_call['step']})")
            if not tool_call['result'].success:
                print(f"     Error: {tool_call['result'].error}")
        
        # Check completion status
        print(f"\n✅ Workflow Complete: {state.is_complete()}")
        
    except Exception as e:
        print(f"❌ Error in workflow monitoring example: {e}")

def example_configuration():
    """Example 5: Configuration management"""
    print("\n🔧 Example 5: Configuration Management")
    print("=" * 50)
    
    try:
        from config.config import Config
        
        # Show current configuration
        print("⚙️  Current Configuration:")
        print(f"  Database Host: {Config.DB_CONFIG['host']}")
        print(f"  Database Port: {Config.DB_CONFIG['port']}")
        print(f"  Database Name: {Config.DB_CONFIG['database']}")
        print(f"  OpenAI Model: {Config.OPENAI_MODEL}")
        print(f"  Max Retries: {Config.MAX_RETRIES}")
        print(f"  Temperature: {Config.TEMPERATURE}")
        print(f"  Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
        
        # Show database URL
        db_url = Config.get_database_url()
        print(f"  Database URL: {db_url}")
        
        # Test configuration validation
        try:
            Config.validate()
            print("✅ Configuration validation passed")
        except Exception as e:
            print(f"⚠️  Configuration validation failed (expected without env vars): {e}")
        
    except Exception as e:
        print(f"❌ Error in configuration example: {e}")

def main():
    """Run all examples"""
    print("🚀 Agentic AI System - Example Usage")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Tools", example_custom_tools),
        ("Error Handling", example_error_handling),
        ("Workflow Monitoring", example_workflow_monitoring),
        ("Configuration", example_configuration)
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example '{name}' failed: {e}")
        
        print("\n" + "-" * 60)
    
    print("\n🎉 All examples completed!")
    print("\n💡 To run the full system:")
    print("   python main.py demo          # Run demo queries")
    print("   python main.py interactive   # Run in interactive mode")
    print("   python test_system.py        # Run system tests")

if __name__ == "__main__":
    main()
