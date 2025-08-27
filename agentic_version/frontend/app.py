#!/usr/bin/env python3
"""
Streamlit Frontend for Agentic AI System
A user-friendly interface for natural language to SQL queries
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the parent directory to Python path to import the agentic system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure plotly for better colors and visibility
import plotly.io as pio
pio.templates.default = "plotly_white"

# Define a nice color palette for visualizations
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



def create_simple_test_chart():
    """Create a simple test chart to verify Plotly functionality"""
    test_data = {
        'Category': ['A', 'B', 'C', 'D', 'E'],
        'Value': [10, 20, 15, 25, 30],
        'Color': ['red', 'blue', 'green', 'orange', 'purple']
    }
    df = pd.DataFrame(test_data)
    
    fig = px.bar(
        df,
        x='Category',
        y='Value',
        color='Color',
        title='Test Chart - Plotly Functionality Verification',
        template='plotly_white',
        color_discrete_map={
            'red': '#ff0000',
            'blue': '#0000ff', 
            'green': '#00ff00',
            'orange': '#ffa500',
            'purple': '#800080'
        }
    )
    
    # Add some styling
    fig.update_layout(
        font=dict(size=14),
        showlegend=True,
        height=500
    )
    
    return fig

# Import the agentic system components
from config.config import Config
from agents.agent_planner import AgentPlanner
from config.state import AgentState

# Page configuration
st.set_page_config(
    page_title="Agentic AI Query System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .tool-result {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    """Initialize the agent planner (cached to avoid re-initialization)"""
    try:
        agent = AgentPlanner()
        return agent, None
    except Exception as e:
        return None, str(e)

def display_tool_results(state: AgentState):
    """Display the results of each tool execution"""
    if not state.tool_calls:
        st.info("No tools have been executed yet.")
        return
    
    st.subheader("🔧 Tool Execution Results")
    
    for i, tool_call in enumerate(state.tool_calls, 1):
        tool_name = tool_call['tool']
        step = tool_call['step']
        result = tool_call['result']
        
        with st.expander(f"Step {i}: {tool_name.title()} ({step})", expanded=True):
            if result.success:
                st.success(f"✅ {tool_name.title()} completed successfully")
                
                # Display tool-specific results
                if tool_name == "keyword_normalizer":
                    data = result.data
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Query:**")
                        st.code(data.get('original_query', ''))
                    with col2:
                        st.write("**Normalized Query:**")
                        st.code(data.get('normalized_query', ''))
                    
                    # replacements = data.get('replacements', [])
                    # if replacements:
                    #     st.write("**Keyword Replacements:**")
                    #     for replacement in replacements:
                    #         st.write(f"• {replacement['original']} → {replacement['normalized']}")
                
                elif tool_name == "schema_inspector":
                    data = result.data
                    st.write("**Database Schema:**")
                    for table_name, table_info in data.items():
                        with st.expander(f"📋 {table_name}"):
                            st.write(f"**Columns:**")
                            for col in table_info.get('columns', []):
                                nullable = "NULL" if col.get('nullable') else "NOT NULL"
                                st.write(f"• {col['name']}: {col['type']} ({nullable})")
                
                elif tool_name == "sql_generator":
                    data = result.data
                    st.write("**Generated SQL:**")
                    st.code(data.get('sql_query', ''), language='sql')
                    
                    # Show validation results
                    validation = data.get('validation', {})
                    if validation:
                        st.success("✅ SQL validation passed")
                    else:
                        st.error("❌ SQL validation failed")
                        # st.write(validation.get('validation_result', ''))
                
                elif tool_name == "db_query":
                    data = result.data
                    if hasattr(data, 'shape'):
                        st.write(f"**Query Results:** {data.shape[0]} rows, {data.shape[1]} columns")
                        if not data.empty:
                            st.dataframe(data.head(10))
                        else:
                            st.warning("No data returned from query")
                    else:
                        st.write("**Query Results:**")
                        st.write(data)
                
                elif tool_name == "result_formatter":
                    data = result.data
                    st.write("**Natural Language Response:**")
                    st.write(data.get('natural_response', ''))
                    
                    # Show visualization code
                    viz_code = data.get('visualization_code', '')
                    if viz_code:
                        st.write("**Visualization Code:**")
                        st.code(viz_code, language='python')
                        
                        # Execute the original visualization code
                        try:
                            # Create a local namespace for execution
                            local_vars = {
                                'pd': pd,
                                'px': px,
                                'go': go,
                                'st': st,
                                'plotly': __import__('plotly'),
                                'plotly_white': 'plotly_white'
                            }
                            
                            # Execute the visualization code
                            exec(viz_code, globals(), local_vars)
                            
                            # Check if a figure was created and display it properly
                            fig = None
                            
                            # First check if 'fig' was created directly
                            if 'fig' in local_vars and local_vars['fig'] is not None:
                                fig = local_vars['fig']
                            # Then check if 'create_chart' function was created
                            elif 'create_chart' in local_vars:
                                # Get the data from the state
                                if hasattr(state, 'sql_execution_result') and state.sql_execution_result is not None:
                                    if hasattr(state.sql_execution_result, 'shape'):
                                        df = state.sql_execution_result
                                        if not df.empty:
                                            # Convert DataFrame to list of dicts for the create_chart function
                                            data_list = df.to_dict('records')
                                            # Call the create_chart function
                                            fig = local_vars['create_chart'](data_list)
                            
                            if fig is not None:
                                # Display the chart using the standard method
                                st.plotly_chart(fig, use_container_width=True)
                                st.success("✅ Visualization displayed successfully!")
                            else:
                                st.info("ℹ️ Visualization code executed but no figure was created")
                                
                        except Exception as viz_error:
                            st.error(f"❌ Visualization code execution failed: {viz_error}")
            else:
                st.error(f"❌ {tool_name.title()} failed")
                st.write(f"Error: {result.error}")

def display_final_results(state: AgentState):
    """Display the final results of the query processing"""
    st.subheader("🎯 Final Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if state.final_response:
            st.write("**Natural Language Response:**")
            st.write(state.final_response)
        else:
            st.warning("No final response generated")
    
    with col2:
        st.write("**Query Statistics:**")
        st.metric("Tools Executed", len(state.tool_calls))
        st.metric("Errors", len(state.errors))
        st.metric("Retry Count", state.retry_count)
        
        if state.generated_sql:
            st.write("**Generated SQL:**")
            st.code(state.generated_sql, language='sql')
    
    # Display visualization if available
    if hasattr(state, 'visualization_code') and state.visualization_code:
        st.markdown("---")
        st.subheader("📊 Data Visualization")
        
        try:
            # Create a local namespace for execution
            local_vars = {
                'pd': pd,
                'px': px,
                'go': go,
                'st': st,
                'plotly': __import__('plotly'),
                'plotly_white': 'plotly_white'
            }
            
            # Execute the visualization code
            exec(state.visualization_code, globals(), local_vars)
            
            # Check if a figure was created and display it properly
            fig = None
            
            # First check if 'fig' was created directly
            if 'fig' in local_vars and local_vars['fig'] is not None:
                fig = local_vars['fig']
            # Then check if 'create_chart' function was created
            elif 'create_chart' in local_vars:
                # Get the data from the state
                if hasattr(state, 'sql_execution_result') and state.sql_execution_result is not None:
                    if hasattr(state.sql_execution_result, 'shape'):
                        df = state.sql_execution_result
                        if not df.empty:
                            # Convert DataFrame to list of dicts for the create_chart function
                            data_list = df.to_dict('records')
                            # Call the create_chart function
                            fig = local_vars['create_chart'](data_list)
            
            if fig is not None:
                # Display the chart using the standard method
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Visualization displayed successfully!")
            else:
                st.info("ℹ️ Visualization code executed but no figure was created")
                
        except Exception as e:
            st.error(f"❌ Visualization execution failed: {e}")
            st.code(state.visualization_code, language='python')

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic AI Query System</h1>', unsafe_allow_html=True)
    st.markdown("Transform natural language queries into SQL and get intelligent insights from your database.")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Initialize agent
    agent, error = initialize_agent()
    
    if error:
        st.error(f"Failed to initialize agent: {error}")
        st.stop()
    
    # Add a button to test Plotly functionality
    if st.sidebar.button("📊 Test Plotly Charts"):
        st.sidebar.write("Testing Plotly functionality...")
        try:
            test_fig = create_simple_test_chart()
            st.plotly_chart(test_fig, use_container_width=True)
            st.sidebar.success("✅ Plotly is working correctly!")
        except Exception as e:
            st.sidebar.error(f"❌ Plotly test failed: {e}")
    
    # Sidebar info
    st.sidebar.info("""
    **How to use:**
    1. Enter your natural language query
    2. Click 'Process Query'
    3. View the results and generated SQL
    4. Explore the tool execution details
    """)
    
    # Example queries
    st.sidebar.subheader("💡 Example Queries")
    example_queries = [
        "Show me data from Riyadh Office",
        "What are the top legal entity types by contributor count?",
        "How many contributors are there in the technology sector?",
        "Show the breakdown of contributors by occupation group",
        "Which cities have the highest number of private entities?",
        "Compare construction and commerce sectors across cities"
    ]
    
    selected_example = st.sidebar.selectbox(
        "Choose an example:",
        ["Custom Query"] + example_queries
    )
    
    # Main content area
    st.subheader("🔍 Query Processing")
    
    # Query input
    if selected_example == "Custom Query":
        user_query = st.text_area(
            "Enter your natural language query:",
            placeholder="e.g., Show me the distribution of contributors by economic activity in Riyadh Office",
            height=100
        )
    else:
        user_query = st.text_area(
            "Query:",
            value=selected_example,
            height=100
        )
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button("�� Process Query", type="primary", use_container_width=True)
    
    # Processing section
    if process_button and user_query.strip():
        st.markdown("---")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Process the query
            status_text.text("🔄 Initializing workflow...")
            progress_bar.progress(10)
            
            status_text.text("🔍 Processing query...")
            progress_bar.progress(30)
            
            result_state = agent.process_query(user_query.strip())
            
            status_text.text("✅ Query processed successfully!")
            progress_bar.progress(100)
            
            # Display results
            st.markdown("---")
            
            # Show final results
            display_final_results(result_state)
            
            # Show detailed tool results
            st.markdown("---")
            display_tool_results(result_state)
            
            # Show errors if any
            if result_state.errors:
                st.markdown("---")
                st.subheader("⚠️ Errors Encountered")
                for i, error in enumerate(result_state.errors, 1):
                    st.error(f"Error {i}: {error}")
            
            # Show completion status
            st.markdown("---")
            if result_state.is_complete():
                st.success("🎉 Query processing completed successfully!")
            else:
                st.warning("⚠️ Query processing encountered issues")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Agentic AI System | Built with Streamlit</p>
        <p>Database: PostgreSQL | LLM: OpenAI GPT-3.5-turbo</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
