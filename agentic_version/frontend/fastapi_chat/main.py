#!/usr/bin/env python3
"""
FastAPI Chat Interface for Agentic AI System
A web-based chat interface using FastAPI and Jinja2 templates
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import shutil
import hashlib
import base64

# Add the parent directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.llm_manager import LLMManager
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from tools.enhanced_pdf_report_generator import create_enhanced_gosi_report

# Add the parent directory to Python path to import the agentic system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the agentic system components
from config.config import Config
from agents.agent_planner import AgentPlanner

# Initialize FastAPI app
app = FastAPI(title="AI Chat Assistant", description="Chat interface for Agentic AI System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Global agent instance
agent: Optional[AgentPlanner] = None

# Global LLM manager for general questions
llm_manager: Optional[LLMManager] = None

# LRU Cache for query storage with max 200 entries
MAX_QUERY_CACHE_SIZE = 2
query_storage: OrderedDict[str, Any] = OrderedDict()


def add_to_query_storage(query_id: str, data: Dict[str, Any]):
    """Add data to query storage with LRU eviction policy"""
    # global query_storage

    # If key already exists, move it to end (most recently used)
    if query_id in query_storage:
        query_storage.move_to_end(query_id)
        query_storage[query_id] = data
    else:
        # Add new entry
        query_storage[query_id] = data

        # If we exceed the limit, remove the least recently used item
        if len(query_storage) > MAX_QUERY_CACHE_SIZE:
            oldest_key = next(iter(query_storage))
            del query_storage[oldest_key]
            print(f"🗑️ Evicted oldest query from cache: {oldest_key}")

    print(f"📊 Query storage now has {len(query_storage)} entries (max: {MAX_QUERY_CACHE_SIZE})")


def get_from_query_storage(query_id: str) -> Optional[Dict[str, Any]]:
    """Get data from query storage and mark as recently used"""
    # global query_storage

    if query_id in query_storage:
        # Move to end to mark as recently used
        query_storage.move_to_end(query_id)
        return query_storage[query_id]
    return None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    message: str
    has_chart: bool = False
    has_report: bool = False
    has_data: bool = False
    data: Optional[List[Dict]] = None
    columns: Optional[List[str]] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    query_id: Optional[str] = None
    error: Optional[str] = None

class ChartResponse(BaseModel):
    success: bool
    chart_data: Optional[List[Dict]] = None
    viz_code: Optional[str] = None
    plot_data: Optional[List[Dict]] = None  # Plotly plot data (list of traces)
    plot_layout: Optional[Dict] = None  # Plotly layout
    plot_config: Optional[Dict] = None  # Plotly config
    error: Optional[str] = None

class ReportResponse(BaseModel):
    success: bool
    report_url: Optional[str] = None
    error: Optional[str] = None

def initialize_agent():
    """Initialize the agent planner"""
    global agent
    try:
        agent = AgentPlanner()
        return True
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return False

def initialize_llm_manager():
    """Initialize the LLM manager for general questions"""
    global llm_manager
    try:
        llm_manager = LLMManager()
        return True
    except Exception as e:
        print(f"Failed to initialize LLM manager: {e}")
        return False

def execute_visualization_code(viz_code: str, data: pd.DataFrame) -> Dict:
    """Execute visualization code and return Plotly plot data"""
    try:
        print("Executing visualization code...")
        
        # Create a simple execution environment
        exec_globals = {
            'pd': pd,
            'px': px,
            'go': go,
            'data': data,
            'Figure': go.Figure,
            'Bar': go.Bar,
            'Scatter': go.Scatter,
            'Pie': go.Pie,
            'Histogram': go.Histogram,
            'Box': go.Box,
            'Violin': go.Violin,
            'Heatmap': go.Heatmap,
            'Surface': go.Surface,
            'Layout': go.Layout
        }
        
        # Execute the code
        exec(viz_code, exec_globals)
        
        # Find the figure - look for common variable names
        fig = None
        for var_name in ['fig', 'figure', 'chart', 'plot', 'graph']:
            if var_name in exec_globals:
                fig = exec_globals[var_name]
                if hasattr(fig, 'to_dict'):
                    print(f"Found figure in variable: {var_name}")
                    break
        
        # If no figure found, create fallback
        if fig is None:
            print("No figure found, creating fallback chart")
            fig = create_fallback_figure(data)
        
        # Convert figure to proper JSON format
        fig_dict = fig.to_dict()
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                # Check if it's a binary buffer format
                if 'bdata' in obj and 'dtype' in obj:
                    try:
                        bdata = base64.b64decode(obj['bdata'])
                        dtype = obj['dtype']
                        arr = np.frombuffer(bdata, dtype=dtype)
                        return arr.tolist()
                    except Exception as e:
                        print(f"Error converting binary buffer: {e}")
                        return obj
                else:
                    return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert the figure data to JSON-serializable format
        converted_data = convert_numpy(fig_dict.get('data', []))
        converted_layout = convert_numpy(fig_dict.get('layout', {}))
        converted_config = convert_numpy(fig_dict.get('config', {}))
        
        result = {
            'data': converted_data,
            'layout': converted_layout,
            'config': converted_config
        }
        
        print(f"Plot data converted successfully: {len(converted_data)} traces")
        if converted_data:
            print(f"First trace type: {converted_data[0].get('type', 'unknown')}")
            print(f"First trace keys: {list(converted_data[0].keys())}")
            # Check for common data fields
            for key in ['x', 'y', 'values', 'labels']:
                if key in converted_data[0]:
                    value = converted_data[0][key]
                    print(f"First trace {key}: {value[:5] if isinstance(value, list) and len(value) > 5 else value}")
        else:
            print("WARNING: No traces found in converted data!")
        
        return result
        
    except Exception as e:
        print(f"Error executing visualization code: {e}")
        # Create fallback
        fig = create_fallback_figure(data)
        fig_dict = fig.to_dict()
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                # Check if it's a binary buffer format
                if 'bdata' in obj and 'dtype' in obj:
                    try:
                        bdata = base64.b64decode(obj['bdata'])
                        dtype = obj['dtype']
                        arr = np.frombuffer(bdata, dtype=dtype)
                        return arr.tolist()
                    except Exception as e:
                        print(f"Error converting binary buffer: {e}")
                        return obj
                else:
                    return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_data = convert_numpy(fig_dict.get('data', []))
        converted_layout = convert_numpy(fig_dict.get('layout', {}))
        converted_config = convert_numpy(fig_dict.get('config', {}))
        
        return {
            'data': converted_data,
            'layout': converted_layout,
            'config': converted_config
        }

def preprocess_visualization_code(viz_code: str) -> str:
    """Pre-process visualization code to handle common issues"""
    lines = viz_code.split('\n')
    processed_lines = []
    
    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            print(f"Skipping import line: {line.strip()}")
            continue
        
        # Skip fig.show() calls
        if 'fig.show()' in line or '.show()' in line:
            print(f"Skipping show() call: {line.strip()}")
            continue
        
        # Skip example data creation (we use the real data)
        if 'data = pd.DataFrame({' in line and 'Example usage' in viz_code:
            print(f"Skipping example data creation: {line.strip()}")
            continue
        
        # Fix common Plotly Figure references
        if 'plotly.Figure' in line:
            line = line.replace('plotly.Figure', 'Figure')
            print(f"Fixed plotly.Figure reference: {line.strip()}")
        
        # Fix common go.Figure references to just Figure
        if 'go.Figure' in line:
            line = line.replace('go.Figure', 'Figure')
            print(f"Fixed go.Figure reference: {line.strip()}")
        
        # Skip empty lines and comments that might cause issues
        if line.strip() == '' or line.strip().startswith('#'):
            processed_lines.append(line)
            continue
        
        processed_lines.append(line)
    
    processed_code = '\n'.join(processed_lines)
    
    # Remove any remaining example data blocks
    if 'Example usage' in processed_code:
        parts = processed_code.split('Example usage')
        processed_code = parts[0].strip()
    
    # Check if there are function definitions but no function calls
    if 'def ' in processed_code and 'fig = ' not in processed_code:
        print("Found function definition but no execution, adding function call")
        # Try to find the main function name
        function_names = []
        for line in processed_code.split('\n'):
            if line.strip().startswith('def '):
                func_name = line.strip().split('(')[0].replace('def ', '').strip()
                function_names.append(func_name)
        
        # Add function calls for each function found
        if function_names:
            processed_code += '\n\n# Execute the function(s) with error handling\n'
            for func_name in function_names:
                if func_name in ['create_chart', 'generate_chart', 'plot_chart', 'visualize']:
                    processed_code += f'try:\n'
                    processed_code += f'    fig = {func_name}(data)\n'
                    processed_code += f'    print(f"Successfully created figure with {func_name}")\n'
                    processed_code += f'except Exception as e:\n'
                    processed_code += f'    print(f"Error in {func_name}: {{e}}")\n'
                    processed_code += f'    fig = None\n'
                    print(f"Added function call with error handling: fig = {func_name}(data)")
                    break
            else:
                # If no standard function name found, use the first one
                processed_code += f'try:\n'
                processed_code += f'    fig = {function_names[0]}(data)\n'
                processed_code += f'    print(f"Successfully created figure with {function_names[0]}")\n'
                processed_code += f'except Exception as e:\n'
                processed_code += f'    print(f"Error in {function_names[0]}: {{e}}")\n'
                processed_code += f'    fig = None\n'
                print(f"Added function call with error handling: fig = {function_names[0]}(data)")
    
    return processed_code

def create_fallback_figure(data: pd.DataFrame) -> go.Figure:
    """Create a simple fallback chart when visualization code fails"""
    try:
        if data.empty or len(data.columns) < 2:
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="No Data Available",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Get first two columns for x and y
        x_col = data.columns[0]
        y_col = data.columns[1]
        
        # Create a simple bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=data[x_col].tolist(),
                y=data[y_col].tolist(),
                name='Data'
            )
        ])
        
        fig.update_layout(
            title='Data Visualization',
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(t=50, r=50, b=50, l=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating fallback chart: {e}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Chart Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

@app.on_event("startup")
async def startup_event():
    """Initialize the agent and LLM manager on startup"""
    print("🚀 Starting AI Chat Assistant...")
    if not initialize_agent():
        print("❌ Failed to initialize agent")
    else:
        print("✅ Agent initialized successfully")
    
    if not initialize_llm_manager():
        print("❌ Failed to initialize LLM manager")
    else:
        print("✅ LLM manager initialized successfully")

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Main chat page"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def process_chat_message(chat_request: ChatRequest):
    """Process a chat message and return the AI response"""
    global agent, llm_manager
    
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    if not llm_manager:
        raise HTTPException(status_code=500, detail="LLM manager not initialized")
    
    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        user_query = chat_request.message.strip()
        
        # Check if this is a general question or data-related query
        is_data_related = llm_manager.is_data_related_query(user_query)
        
        if not is_data_related:
            # Handle as general conversation
            print(f"🤖 Processing general question: {user_query}")
            general_response = llm_manager.handle_general_question(user_query)
            
            if general_response['success']:
                return ChatResponse(
                    success=True,
                    message=general_response['response'],
                    has_chart=False,
                    has_report=False,
                    has_data=False,
                    data=None,
                    columns=None,
                    row_count=None,
                    column_count=None,
                    query_id=None
                )
            else:
                return ChatResponse(
                    success=False,
                    message="I'm sorry, I'm having trouble processing your question right now. Please try again.",
                    has_chart=False,
                    has_report=False,
                    has_data=False,
                    data=None,
                    columns=None,
                    row_count=None,
                    column_count=None,
                    query_id=None
                )
        
        # Process the query using basic workflow (without chart generation)
        print(f"📊 Processing data-related query: {user_query}")
        result_state = agent.process_query_basic(user_query)
        
        # Get the natural language response
        response = result_state.final_response or "I processed your query but couldn't generate a response."

        message_hash = hashlib.md5(chat_request.message.strip().encode()).hexdigest()[:8]
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{message_hash}"
        print(f"🆔 Generated query_id: {query_id}")
        
        # Check if chart and report data are available
        has_data = (hasattr(result_state, 'sql_execution_result') and 
                   result_state.sql_execution_result is not None and
                   not result_state.sql_execution_result.empty)
        
        has_chart = has_data  # If we have data, we can generate charts
        has_report = has_data  # If we have data, we can generate reports
        
        # Prepare dataframe data if available
        dataframe_data = None
        columns = None
        row_count = None
        column_count = None
        
        if has_data and hasattr(result_state, 'sql_execution_result') and result_state.sql_execution_result is not None:
            dataframe_data = result_state.sql_execution_result.to_dict('records')
            columns = list(result_state.sql_execution_result.columns)
            row_count = len(dataframe_data)
            column_count = len(columns)
        
        # Store the result state for later use (always store if we have data)
        if has_data:
            storage_data= {
                'result_state': result_state,
                'original_query': chat_request.message.strip(),
                'timestamp': datetime.now().isoformat()
            }
            add_to_query_storage(query_id, storage_data)
            print(f"✅ Stored query in storage with ID: {query_id}")
        else:
            print(f"⚠️ No data available, not storing query: {query_id}")
        
        return ChatResponse(
            success=True,
            message=response,
            has_chart=bool(has_chart),
            has_report=bool(has_report),
            has_data=bool(has_data),
            data=dataframe_data,
            columns=columns,
            row_count=row_count,
            column_count=column_count,
            query_id=query_id if (has_chart or has_report or has_data) else None
        )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            message="",
            has_chart=False,
            has_report=False,
            has_data=False,
            data=None,
            columns=None,
            row_count=None,
            column_count=None,
            query_id=None,
            error=f"Error processing query: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent_initialized": agent is not None}

@app.post("/api/chart/{query_id}", response_model=ChartResponse)
async def generate_chart(query_id: str):
    """Generate chart for a specific query"""
    
    stored_data = get_from_query_storage(query_id)
    if not stored_data:
        print(f"❌ Query ID {query_id} not found in storage")
        raise HTTPException(status_code=404, detail=f"Query not found. Available queries: {list(query_storage.keys())}")
    
    try:
        result_state = stored_data['result_state']
        original_query = stored_data['original_query']
        
        if (hasattr(result_state, 'sql_execution_result') and 
            result_state.sql_execution_result is not None and
            not result_state.sql_execution_result.empty):
            
            # Generate chart using the new workflow with existing data
            print(f"Generating chart for query: {original_query}")
            print(f"Data shape: {result_state.sql_execution_result.shape}")
            print(f"Database schemas available: {bool(result_state.database_schemas)}")
            
            chart_result_state = agent.generate_chart_for_data(
                original_query,
                result_state.sql_execution_result,
                result_state.database_schemas
            )
            
            print(f"Chart generation result: {hasattr(chart_result_state, 'visualization_code')}")
            if hasattr(chart_result_state, 'visualization_code'):
                print(f"Visualization code length: {len(chart_result_state.visualization_code) if chart_result_state.visualization_code else 0}")
            
            if (hasattr(chart_result_state, 'visualization_code') and 
                chart_result_state.visualization_code):
                
                viz_code = chart_result_state.visualization_code
                chart_data = result_state.sql_execution_result.to_dict('records')
                
                # Execute the visualization code to generate plot data
                print(f"Executing visualization code to generate plot data...")
                plot_result = execute_visualization_code(viz_code, result_state.sql_execution_result)
                print(f"Plot result generated: {type(plot_result)}")
                print(f"Plot result keys: {plot_result.keys() if isinstance(plot_result, dict) else 'Not a dict'}")
                
                plot_data = plot_result.get('data')
                plot_layout = plot_result.get('layout')
                plot_config = plot_result.get('config')
                
                print(f"Plot data type: {type(plot_data)}, length: {len(plot_data) if plot_data else 0}")
                print(f"Plot layout type: {type(plot_layout)}")
                print(f"Plot config type: {type(plot_config)}")
                
                if plot_data:
                    print(f"First trace in plot_data: {plot_data[0] if plot_data else 'No data'}")
                
                return ChartResponse(
                    success=True,
                    chart_data=chart_data,
                    viz_code=viz_code,
                    plot_data=plot_data,
                    plot_layout=plot_layout,
                    plot_config=plot_config
                )
            else:
                return ChartResponse(
                    success=False,
                    error="Chart generation failed - no visualization code generated"
                )
        else:
            return ChartResponse(
                success=False,
                error="No data available for chart generation"
            )
            
    except Exception as e:
        return ChartResponse(
            success=False,
            error=f"Error generating chart: {str(e)}"
        )


@app.post("/api/report/{query_id}")
async def generate_report(query_id: str):
    """Generate PDF report for a specific query and return the file"""
    
    stored_data = get_from_query_storage(query_id)
    if not stored_data:
        raise HTTPException(status_code=404, detail="Query not found")
    
    try:
        result_state = stored_data['result_state']
        original_query = stored_data['original_query']
        
        if (hasattr(result_state, 'sql_execution_result') and 
            result_state.sql_execution_result is not None and
            not result_state.sql_execution_result.empty):
            
            # Generate enhanced PDF report (without technical details)
            pdf_path = create_enhanced_gosi_report(
                query=original_query,
                data=result_state.sql_execution_result,
                description=result_state.final_response or "No description available",
                fig=None,
                language='auto'  # Auto-detect language from query
            )
            
            # Create reports directory if it doesn't exist
            reports_dir = "static/reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Copy the generated PDF to the static reports directory
            filename = f"report_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            static_pdf_path = os.path.join(reports_dir, filename)
            shutil.copy2(pdf_path, static_pdf_path)
            
            # Clean up the original temporary file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            
            # Return the file for download
            return FileResponse(
                path=static_pdf_path,
                filename=filename,
                media_type='application/pdf',
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            raise HTTPException(status_code=400, detail="No data available for report generation")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/api/examples")
async def get_example_queries():
    """Get example queries for the chat interface"""
    examples = [
        "Compare private vs stock contributors in Riyadh for whole year of 2018?",
        "Compare construction and commerce sectors across top three cities?",
        "compare contributor in manufacturing and community service sector in 2018 in riyadh for each quarter?"
    ]
    return {"examples": examples}

if __name__ == "__main__":
    print("🚀 Starting FastAPI Chat Server...")
    print("📱 Chat interface will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
