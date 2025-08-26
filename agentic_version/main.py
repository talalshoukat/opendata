#!/usr/bin/env python3
"""
Agentic AI System for Natural Language to SQL Queries
A robust, multi-step agent system using LangGraph for intelligent query processing.

This system demonstrates:
1. Modular tool-based architecture
2. Intelligent workflow planning and execution
3. Error handling and retry mechanisms
4. Dynamic tool selection and execution
"""

import logging
import sys
from typing import Dict, Any
from agents.agent_planner import AgentPlanner
from config.config import Config
from config.state import AgentState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentic_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_separator(title: str):
    """Print a formatted separator with title"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)

def print_workflow_status(state: AgentState):
    """Print the current workflow status"""
    print(f"\n🔄 Workflow Status: {state.current_step}")
    print(f"📝 User Query: {state.user_query}")
    
    if state.normalized_query and state.normalized_query != state.user_query:
        print(f"🔧 Normalized Query: {state.normalized_query}")
    
    if state.generated_sql:
        print(f"💻 Generated SQL: {state.generated_sql}")
    
    if state.sql_execution_result is not None:
        if hasattr(state.sql_execution_result, 'shape'):
            print(f"📊 Query Results: {state.sql_execution_result.shape[0]} rows, {state.sql_execution_result.shape[1]} columns")
        else:
            print(f"📊 Query Results: {len(state.sql_execution_result) if hasattr(state.sql_execution_result, '__len__') else 'N/A'}")
    
    if state.final_response:
        print(f"💬 Natural Language Response: {state.final_response[:200]}...")
    
    if state.visualization_code:
        print(f"📈 Visualization Code Generated: {len(state.visualization_code)} characters")
    
    if state.errors:
        print(f"❌ Errors ({len(state.errors)}):")
        for i, error in enumerate(state.errors, 1):
            print(f"   {i}. {error}")
    
    print(f"🔄 Retry Count: {state.retry_count}/{state.max_retries}")
    print(f"✅ Workflow Complete: {state.is_complete()}")

def print_tool_execution_summary(state: AgentState):
    """Print a summary of tool executions"""
    if not state.tool_calls:
        print("\n📋 No tools executed yet")
        return
    
    print(f"\n📋 Tool Execution Summary ({len(state.tool_calls)} tools):")
    for i, tool_call in enumerate(state.tool_calls, 1):
        tool_name = tool_call['tool']
        step = tool_call['step']
        success = tool_call['result'].success
        
        status = "✅" if success else "❌"
        print(f"  {i}. {status} {tool_name} ({step})")
        
        if not success and tool_call['result'].error:
            print(f"     Error: {tool_call['result'].error}")

def run_example_queries():
    """Run example queries to demonstrate the system"""
    example_queries = [
        # "Show me the distribution of contributors by economic activity in 'Riyadh Office' ",
        # "What are the top 10 legal entity types by contributor count?",
        # "How many contributors are there in the technology sector?",
        # "Show the breakdown of contributors by occupation group"
    ]
    
    print_separator("AGENTIC AI SYSTEM DEMONSTRATION")
    print("This system demonstrates intelligent query processing with:")
    print("• Keyword normalization using vector similarity")
    print("• Dynamic SQL generation with LLM")
    print("• Intelligent error handling and retry mechanisms")
    print("• Natural language result explanation")
    print("• Automated visualization code generation")
    
    # Initialize the agent planner
    try:
        print("\n🚀 Initializing Agent Planner...")
        agent_planner = AgentPlanner()
        print("✅ Agent Planner initialized successfully!")
        
        # Process each example query
        for i, query in enumerate(example_queries, 1):
            print_separator(f"EXAMPLE QUERY {i}")
            print(f"Query: {query}")
            
            try:
                # Process the query
                print(f"\n🔄 Processing query...")
                result_state = agent_planner.process_query(query)
                
                # Print results
                print_workflow_status(result_state)
                print_tool_execution_summary(result_state)
                
                if result_state.is_complete():
                    print(f"\n🎉 Query {i} processed successfully!")
                else:
                    print(f"\n⚠️  Query {i} encountered issues")
                
            except Exception as e:
                print(f"❌ Error processing query {i}: {e}")
                logger.error(f"Error processing query {i}: {e}")
            
            print("\n" + "-"*60)
        
        # Clean up
        agent_planner.close()
        print("✅ Agent Planner resources cleaned up")
        
    except Exception as e:
        print(f"❌ Error initializing Agent Planner: {e}")
        logger.error(f"Error initializing Agent Planner: {e}")
        return

def interactive_mode():
    """Run the system in interactive mode"""
    print_separator("INTERACTIVE MODE")
    print("Enter your natural language queries (type 'quit' to exit)")
    
    try:
        agent_planner = AgentPlanner()
        print("✅ Agent Planner initialized successfully!")
        
        while True:
            try:
                # Get user input
                user_query = input("\n🤔 Enter your query: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_query:
                    print("Please enter a valid query")
                    continue
                
                print(f"\n🔄 Processing: {user_query}")
                
                # Process the query
                result_state = agent_planner.process_query(user_query)
                
                # Print results
                print_workflow_status(result_state)
                print_tool_execution_summary(result_state)
                
                if result_state.is_complete():
                    print(f"\n🎉 Query processed successfully!")
                    
                    # Show final response
                    if result_state.final_response:
                        print(f"\n💬 Response: {result_state.final_response}")
                    
                    # Show visualization code if available
                    if result_state.visualization_code:
                        print(f"\n📈 Visualization Code:")
                        print("```python")
                        print(result_state.visualization_code)
                        print("```")
                else:
                    print(f"\n⚠️  Query encountered issues")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Error in interactive mode: {e}")
        
        # Clean up
        agent_planner.close()
        print("✅ Agent Planner resources cleaned up")
        
    except Exception as e:
        print(f"❌ Error initializing Agent Planner: {e}")
        logger.error(f"Error initializing Agent Planner: {e}")

def main():
    """Main entry point"""
    try:
        # Validate configuration
        Config.validate()
        print("✅ Configuration validated successfully")
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "demo":
                run_example_queries()
            elif sys.argv[1] == "interactive":
                interactive_mode()
            else:
                print("Usage: python main.py [demo|interactive]")
                print("  demo: Run example queries")
                print("  interactive: Run in interactive mode")
                print("  (no args): Run example queries")
                run_example_queries()
        else:
            run_example_queries()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
