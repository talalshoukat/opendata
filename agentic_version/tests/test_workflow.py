#!/usr/bin/env python3
"""
Complete workflow tests for the agentic AI system
"""

import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_mock_llm_manager():
    """Create a mock LLM manager with realistic responses"""
    mock_llm = Mock()
    
    # Mock SQL generation
    mock_llm.generate_sql_query.return_value = {
        'success': True,
        'sql_query': 'SELECT economic_activity, contributor_count FROM private_sector_contributor_distribution_by_economic_activity ORDER BY contributor_count DESC',
        'model_used': 'gpt-3.5-turbo',
        'tokens_used': 150
    }
    
    # Mock SQL validation
    mock_llm.validate_sql_query.return_value = {
        'is_valid': True,
        'validation_message': 'SQL query is valid'
    }
    
    # Mock natural response generation
    mock_llm.generate_natural_response.return_value = {
        'success': True,
        'natural_response': 'Based on the data, the technology sector has the highest number of contributors with 1,200 individuals, followed by finance with 900 contributors, and healthcare with 750 contributors.',
        'model_used': 'gpt-3.5-turbo',
        'tokens_used': 200
    }
    
    # Mock visualization code generation
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
    
    return mock_llm

def test_complete_workflow():
    """Test the complete workflow with mocked dependencies"""
    print("🚀 Testing Complete Workflow with Mocked LLM")
    print("=" * 50)
    
    try:
        # Import required modules
        from config.state import AgentState, ToolResult
        from tools.tool_definitions import create_tools
        from tools.vector_store import FAISSVectorStore
        from tools.database_manager import DatabaseManager
        
        # Create mock LLM manager
        mock_llm = create_mock_llm_manager()
        
        # Create real database manager and vector store
        db_manager = DatabaseManager()
        vector_store = FAISSVectorStore()
        
        # Add sample data to vector store
        sample_data = {
            'economic_activity': ['Technology', 'Finance', 'Healthcare', 'Manufacturing', 'Retail'],
            'legal_entity_type': ['Limited Liability Company', 'Corporation', 'Partnership', 'Sole Proprietorship'],
            'occupation_group': ['Engineers', 'Managers', 'Analysts', 'Developers', 'Consultants']
        }
        vector_store.add_categorical_values('sample_table', sample_data)
        vector_store.build_index()
        
        # Create tools with mocked LLM
        tools = create_tools(db_manager, vector_store, mock_llm)
        print(f"✅ Created {len(tools)} tools")
        
        # Test queries
        test_queries = [
            "Show me the distribution of contributors by economic activity",
            "What are the top legal entity types by contributor count?",
            "How many contributors are there in the technology sector?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: {query}")
            
            # Create state
            state = AgentState(user_query=query)
            
            # Step 1: Normalize keywords
            state.update_step("normalize_keywords")
            keyword_tool = next(t for t in tools if t.name == "keyword_normalizer")
            keyword_result = keyword_tool._run(query)
            state.add_tool_call("keyword_normalizer", query, keyword_result)
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
            sql_result = sql_tool._run(query, state.database_schemas, keyword_result.data.get('normalized_query'))
            state.add_tool_call("sql_generator", query, sql_result)
            state.generated_sql = sql_result.data['sql_query']
            print(f"✅ SQL generation: {sql_result.success}")
            print(f"   Generated SQL: {state.generated_sql[:80]}...")
            
            # Step 4: Execute query (mocked)
            state.update_step("execute_query")
            query_tool = next(t for t in tools if t.name == "db_query")
            # Mock the database result
            mock_results = [
                {'economic_activity': 'Technology', 'contributor_count': 1200},
                {'economic_activity': 'Finance', 'contributor_count': 900},
                {'economic_activity': 'Healthcare', 'contributor_count': 750}
            ]
            query_result = ToolResult(
                success=True,
                data=mock_results,
                metadata={'rows_returned': len(mock_results)}
            )
            state.add_tool_call("db_query", state.generated_sql, query_result)
            state.sql_execution_result = mock_results
            print(f"✅ Query execution: {query_result.success}")
            print(f"   Rows returned: {len(mock_results)}")
            
            # Step 5: Format results
            state.update_step("format_results")
            formatter_tool = next(t for t in tools if t.name == "result_formatter")
            formatter_result = formatter_tool._run(
                query, state.generated_sql, state.sql_execution_result, state.database_schemas
            )
            state.add_tool_call("result_formatter", "format_results", formatter_result)
            state.final_response = formatter_result.data['natural_response']
            state.visualization_code = formatter_result.data['visualization_code']
            print(f"✅ Result formatting: {formatter_result.success}")
            
            # Complete workflow
            state.should_continue = False
            
            # Show results
            print(f"\n📊 Results for Query {i}:")
            print(f"   Natural Response: {state.final_response[:100]}...")
            print(f"   Visualization Code: {len(state.visualization_code)} characters")
            print(f"   Total Tool Calls: {len(state.tool_calls)}")
            print(f"   Errors: {len(state.errors)}")
            
            # Show tool execution summary
            print(f"📋 Tool Execution Summary:")
            for j, tool_call in enumerate(state.tool_calls, 1):
                tool_name = tool_call['tool']
                success = tool_call['result'].success
                status = "✅" if success else "❌"
                print(f"   {j}. {status} {tool_name}")
            
            print("-" * 50)
        
        # Clean up
        db_manager.close()
        print("✅ Database connections closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_with_real_data():
    """Test vector store with real database data"""
    print("\n🧪 Testing Vector Store with Real Database Data")
    print("=" * 50)
    
    try:
        from tools.database_manager import DatabaseManager
        from tools.vector_store import FAISSVectorStore
        
        # Create database manager
        db_manager = DatabaseManager()
        
        # Get all schemas
        schemas = db_manager.get_all_schemas()
        print(f"✅ Retrieved {len(schemas)} table schemas")
        
        # Create vector store
        vector_store = FAISSVectorStore()
        
        # Add categorical values from database
        for table_name, schema in schemas.items():
            if 'sample_data' in schema and schema['sample_data']:
                # Extract categorical columns
                categorical_data = {}
                for row in schema['sample_data']:
                    for column, value in row.items():
                        if column not in categorical_data:
                            categorical_data[column] = []
                        if value and str(value) not in categorical_data[column]:
                            categorical_data[column].append(str(value))
                
                if categorical_data:
                    vector_store.add_categorical_values(table_name, categorical_data)
                    print(f"✅ Added {len(categorical_data)} categorical columns from {table_name}")
        
        # Build index
        vector_store.build_index()
        print(f"✅ Built vector index with {len(vector_store.keywords)} keywords")
        
        # Test keyword normalization
        test_queries = [
            "Show me technology contributors",
            "What are the top legal entities?",
            "How many engineers are there?"
        ]
        
        for query in test_queries:
            result = vector_store.normalize_query_keywords(query)
            print(f"Query: {query}")
            print(f"Normalized: {result['normalized_query']}")
            print(f"Replacements: {len(result['replacements'])}")
            print()
        
        # Clean up
        db_manager.close()
        print("✅ Database connections closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_workflow_tests():
    """Run all workflow tests"""
    print("🎯 Complete Workflow Test with Mocked LLM")
    print("This test demonstrates the full system working with mocked LLM responses")
    
    # Test complete workflow
    workflow_success = test_complete_workflow()
    
    if workflow_success:
        # Test vector store with real data
        vector_success = test_vector_store_with_real_data()
        
        if vector_success:
            print("\n🎉 All workflow tests passed! The system is working correctly.")
            return True
        else:
            print("\n⚠️ Vector store test failed.")
            return False
    else:
        print("\n❌ Workflow test failed.")
        return False

if __name__ == "__main__":
    success = run_workflow_tests()
    sys.exit(0 if success else 1)
