#!/usr/bin/env python3
"""
Demo script showing the agentic AI system working with mocked dependencies
"""

import sys
import os
from unittest.mock import Mock, patch
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_mock_dependencies():
    """Create mock dependencies for the demo"""
    
    # Mock database manager
    mock_db = Mock()
    mock_db.get_all_schemas.return_value = {
        'private_sector_contributor_distribution_by_legal_entity': {
            'columns': ['legal_entity_type', 'contributor_count', 'percentage'],
            'sample_data': [
                {'legal_entity_type': 'Limited Liability Company', 'contributor_count': 1500, 'percentage': 45.2},
                {'legal_entity_type': 'Corporation', 'contributor_count': 800, 'percentage': 24.1},
                {'legal_entity_type': 'Partnership', 'contributor_count': 600, 'percentage': 18.1}
            ]
        },
        'private_sector_contributor_distribution_by_economic_activity': {
            'columns': ['economic_activity', 'contributor_count', 'percentage'],
            'sample_data': [
                {'economic_activity': 'Technology', 'contributor_count': 1200, 'percentage': 36.2},
                {'economic_activity': 'Finance', 'contributor_count': 900, 'percentage': 27.1},
                {'economic_activity': 'Healthcare', 'contributor_count': 750, 'percentage': 22.6}
            ]
        }
    }
    mock_db.execute_query.return_value = [
        {'economic_activity': 'Technology', 'contributor_count': 1200, 'percentage': 36.2},
        {'economic_activity': 'Finance', 'contributor_count': 900, 'percentage': 27.1},
        {'economic_activity': 'Healthcare', 'contributor_count': 750, 'percentage': 22.6}
    ]
    
    # Mock LLM manager
    mock_llm = Mock()
    mock_llm.generate_sql_query.return_value = {
        'success': True,
        'sql_query': 'SELECT economic_activity, contributor_count FROM private_sector_contributor_distribution_by_economic_activity ORDER BY contributor_count DESC',
        'model_used': 'gpt-3.5-turbo',
        'tokens_used': 150
    }
    mock_llm.validate_sql_query.return_value = {
        'is_valid': True,
        'validation_message': 'SQL query is valid'
    }
    mock_llm.generate_natural_response.return_value = {
        'success': True,
        'natural_response': 'Based on the data, the technology sector has the highest number of contributors with 1,200 individuals, followed by finance with 900 contributors, and healthcare with 750 contributors.',
        'model_used': 'gpt-3.5-turbo',
        'tokens_used': 200
    }
    mock_llm.generate_visualization_code.return_value = {
        'success': True,
        'visualization_code': '''
import plotly.express as px
import pandas as pd

# Create sample data
data = [
    {"economic_activity": "Technology", "contributor_count": 1200},
    {"economic_activity": "Finance", "contributor_count": 900},
    {"economic_activity": "Healthcare", "contributor_count": 750}
]
df = pd.DataFrame(data)

# Create bar chart
fig = px.bar(df, x='economic_activity', y='contributor_count', 
             title='Contributors by Economic Activity',
             color='contributor_count')
fig.show()
        ''',
        'tokens_used': 300
    }
    
    return mock_db, mock_llm

def demo_workflow():
    """Demonstrate the complete workflow"""
    print("🚀 Agentic AI System Demo")
    print("=" * 50)
    
    try:
        # Import required modules
        from config.state import AgentState, ToolResult
        from tools.tool_definitions import create_tools
        from tools.vector_store import FAISSVectorStore
        
        # Create mock dependencies
        mock_db, mock_llm = create_mock_dependencies()
        
        # Create vector store
        vector_store = FAISSVectorStore()
        
        # Add some sample data to vector store
        sample_data = {
            'economic_activity': ['Technology', 'Finance', 'Healthcare', 'Manufacturing', 'Retail'],
            'legal_entity_type': ['Limited Liability Company', 'Corporation', 'Partnership', 'Sole Proprietorship']
        }
        vector_store.add_categorical_values('sample_table', sample_data)
        vector_store.build_index()
        
        # Create tools
        tools = create_tools(mock_db, vector_store, mock_llm)
        print(f"✅ Created {len(tools)} tools")
        
        # Test query
        test_query = "Show me the distribution of contributors by economic activity"
        print(f"\n📝 Test Query: {test_query}")
        
        # Create state
        state = AgentState(user_query=test_query)
        print(f"✅ Initialized state for query")
        
        # Simulate workflow steps
        print("\n🔄 Simulating workflow steps...")
        
        # Step 1: Normalize keywords
        state.update_step("normalize_keywords")
        keyword_tool = next(t for t in tools if t.name == "keyword_normalizer")
        keyword_result = keyword_tool._run(test_query)
        state.add_tool_call("keyword_normalizer", test_query, keyword_result)
        print(f"✅ Keyword normalization: {keyword_result.success}")
        
        # Step 2: Inspect schema
        state.update_step("inspect_schema")
        schema_tool = next(t for t in tools if t.name == "schema_inspector")
        schema_result = schema_tool._run()
        state.add_tool_call("schema_inspector", None, schema_result)
        state.database_schemas = schema_result.data
        print(f"✅ Schema inspection: {schema_result.success}")
        
        # Step 3: Generate SQL
        state.update_step("generate_sql")
        sql_tool = next(t for t in tools if t.name == "sql_generator")
        sql_result = sql_tool._run(test_query, state.database_schemas, keyword_result.data.get('normalized_query'))
        state.add_tool_call("sql_generator", test_query, sql_result)
        state.generated_sql = sql_result.data['sql_query']
        print(f"✅ SQL generation: {sql_result.success}")
        print(f"   Generated SQL: {state.generated_sql[:100]}...")
        
        # Step 4: Execute query
        state.update_step("execute_query")
        query_tool = next(t for t in tools if t.name == "db_query")
        query_result = query_tool._run(state.generated_sql)
        state.add_tool_call("db_query", state.generated_sql, query_result)
        state.sql_execution_result = query_result.data
        print(f"✅ Query execution: {query_result.success}")
        print(f"   Rows returned: {len(query_result.data)}")
        
        # Step 5: Format results
        state.update_step("format_results")
        formatter_tool = next(t for t in tools if t.name == "result_formatter")
        formatter_result = formatter_tool._run(
            test_query, state.generated_sql, state.sql_execution_result, state.database_schemas
        )
        state.add_tool_call("result_formatter", "format_results", formatter_result)
        state.final_response = formatter_result.data['natural_response']
        state.visualization_code = formatter_result.data['visualization_code']
        print(f"✅ Result formatting: {formatter_result.success}")
        
        # Complete workflow
        state.should_continue = False
        print(f"\n🎉 Workflow completed successfully!")
        
        # Show results
        print(f"\n📊 Final Results:")
        print(f"   Natural Language Response: {state.final_response}")
        print(f"   Visualization Code Length: {len(state.visualization_code)} characters")
        print(f"   Total Tool Calls: {len(state.tool_calls)}")
        print(f"   Errors: {len(state.errors)}")
        
        # Show tool execution summary
        print(f"\n📋 Tool Execution Summary:")
        for i, tool_call in enumerate(state.tool_calls, 1):
            tool_name = tool_call['tool']
            success = tool_call['result'].success
            status = "✅" if success else "❌"
            print(f"   {i}. {status} {tool_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_interactive():
    """Interactive demo with user input"""
    print("\n🎮 Interactive Demo")
    print("=" * 30)
    print("Enter queries to test the system (type 'quit' to exit)")
    
    try:
        from config.state import AgentState
        from tools.tool_definitions import create_tools
        from tools.vector_store import FAISSVectorStore
        
        # Setup dependencies
        mock_db, mock_llm = create_mock_dependencies()
        vector_store = FAISSVectorStore()
        tools = create_tools(mock_db, vector_store, mock_llm)
        
        while True:
            query = input("\n🤔 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                print("Please enter a valid query")
                continue
            
            print(f"\n🔄 Processing: {query}")
            
            # Create state and simulate workflow
            state = AgentState(user_query=query)
            
            # Quick simulation
            state.update_step("normalize_keywords")
            keyword_tool = next(t for t in tools if t.name == "keyword_normalizer")
            keyword_result = keyword_tool._run(query)
            state.add_tool_call("keyword_normalizer", query, keyword_result)
            
            state.update_step("generate_sql")
            sql_tool = next(t for t in tools if t.name == "sql_generator")
            sql_result = sql_tool._run(query, mock_db.get_all_schemas(), keyword_result.data.get('normalized_query'))
            state.add_tool_call("sql_generator", query, sql_result)
            
            if sql_result.success:
                print(f"✅ SQL Generated: {sql_result.data['sql_query'][:100]}...")
                
                # Simulate execution and formatting
                state.update_step("execute_query")
                query_tool = next(t for t in tools if t.name == "db_query")
                query_result = query_tool._run(sql_result.data['sql_query'])
                state.add_tool_call("db_query", sql_result.data['sql_query'], query_result)
                
                state.update_step("format_results")
                formatter_tool = next(t for t in tools if t.name == "result_formatter")
                formatter_result = formatter_tool._run(query, sql_result.data['sql_query'], query_result.data, mock_db.get_all_schemas())
                state.add_tool_call("result_formatter", "format_results", formatter_result)
                
                if formatter_result.success:
                    print(f"✅ Response: {formatter_result.data['natural_response']}")
                else:
                    print(f"❌ Formatting failed: {formatter_result.error}")
            else:
                print(f"❌ SQL generation failed: {sql_result.error}")
            
            print(f"📊 Tools executed: {len(state.tool_calls)}")
        
        print("👋 Goodbye!")
        return True
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")
        return False

def main():
    """Main demo function"""
    print("🎯 Agentic AI System - Natural Language to SQL Demo")
    print("This demo shows the complete workflow with mocked external dependencies")
    
    # Run basic workflow demo
    if demo_workflow():
        print("\n" + "="*50)
        
        # Ask if user wants interactive demo
        try:
            choice = input("\n🎮 Would you like to try the interactive demo? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                demo_interactive()
        except KeyboardInterrupt:
            print("\n👋 Demo ended by user")
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    main()
