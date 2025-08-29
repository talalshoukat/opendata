from typing import Dict, List, Any, Optional, Tuple
from langgraph.graph import StateGraph, END
from langchain.tools import BaseTool
from config.state import AgentState, ToolResult
from tools.tool_definitions import create_tools
from tools.database_manager import DatabaseManager
from tools.vector_store import FAISSVectorStore
from tools.llm_manager import LLMManager
import logging
import os
import time

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Simple tool executor for the agent system"""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
    
    def invoke(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        tool_name = tool_call.get("tool")
        tool_input = tool_call.get("tool_input", {})
        
        if tool_name not in self.tools:
            return {
                "output": ToolResult(
                    success=False,
                    data=None,
                    error=f"Tool {tool_name} not found"
                ).dict()
            }
        
        try:
            tool = self.tools[tool_name]
            result = tool._run(**tool_input)
            return {"output": result.dict()}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "output": ToolResult(
                    success=False,
                    data=None,
                    error=str(e)
                ).dict()
            }

class AgentPlanner:
    """Main agent planner that orchestrates the workflow using LangGraph"""
    
    def __init__(self):
        """Initialize the agent planner with all required components"""
        # Initialize core components
        self.db_manager = DatabaseManager()
        self.vector_store = FAISSVectorStore()
        self.llm_manager = LLMManager()
        
        # Check if vector store needs initialization
        self._ensure_vector_store_initialized()
        
        # Create tools
        self.tools = create_tools(self.db_manager, self.vector_store, self.llm_manager)
        self.tool_executor = ToolExecutor(self.tools)
        
        # Build the workflow graphs
        self.workflow = self._build_workflow()
        self.basic_workflow = self._build_basic_workflow()
        self.chart_workflow = self._build_chart_workflow()
        
        logger.info("Agent planner initialized successfully")
    
    def _ensure_vector_store_initialized(self):
        """Ensure vector store is initialized with data from database"""
        try:
            # Check if vector store has any keywords
            stats = self.vector_store.get_collection_stats()
            
            if stats['total_keywords'] == 0 or not stats['is_fitted']:
                logger.info("Vector store is empty or not fitted, initializing with database data...")
                success = self._initialize_vector_store()
                if success:
                    logger.info("Vector store initialized successfully")
                else:
                    logger.warning("Vector store initialization failed, continuing with empty store")
            else:
                logger.info(f"Vector store already initialized with {stats['total_keywords']} keywords")
                
        except Exception as e:
            logger.error(f"Error checking vector store initialization: {e}")
    
    def _initialize_vector_store(self):
        """Initialize vector store with database data"""
        try:
            from config.config import Config
            
            logger.info("🚀 Initializing Vector Store with Database Data")
            
            # Get all schemas
            logger.info("📋 Retrieving database schemas...")
            schemas = self.db_manager.get_all_schemas()
            logger.info(f"✅ Retrieved {len(schemas)} table schemas")
            
            # Add categorical values from database
            total_keywords = 0
            for table_name, schema in schemas.items():
                logger.info(f"📝 Processing table: {table_name}")
                
                # Get actual data from the table
                sample_data = self.db_manager.get_sample_data(table_name, limit=100)
                
                if not sample_data.empty:
                    # Extract only categorical columns (exclude numeric, ID, and date columns)
                    categorical_data = {}
                    for column in sample_data.columns:
                        # Skip columns that are likely not categorical
                        if any(skip in column.lower() for skip in ['id', 'count', 'number', 'amount', 'percentage', 'ratio', 'total', 'sum', 'avg', 'min', 'max']):
                            continue
                        
                        # Skip columns that are mostly numeric
                        numeric_count = 0
                        total_count = 0
                        unique_values = sample_data[column].dropna().unique()
                        
                        for val in unique_values:
                            total_count += 1
                            try:
                                float(str(val))
                                numeric_count += 1
                            except ValueError:
                                pass
                        
                        # Only include columns that are less than 50% numeric and have reasonable number of unique values
                        if total_count > 0 and (numeric_count / total_count) < 0.5 and len(unique_values) > 1 and len(unique_values) < 100:
                            # Convert to strings and filter out empty values
                            string_values = [str(val).strip() for val in unique_values if str(val).strip()]
                            if string_values:
                                categorical_data[column] = string_values
                    
                    if categorical_data:
                        # Add to vector store
                        self.vector_store.add_categorical_values(table_name, categorical_data)
                        
                        # Count keywords
                        table_keywords = sum(len(values) for values in categorical_data.values())
                        total_keywords += table_keywords
                        
                        logger.info(f"   ✅ Added {len(categorical_data)} categorical columns with {table_keywords} unique values")
                    else:
                        logger.info(f"   ⚠️  No categorical data extracted from {table_name}")
                else:
                    logger.info(f"   ⚠️  No data found in {table_name}")
            
            # Build index
            logger.info("🔨 Building vector index...")
            self.vector_store.build_index()
            logger.info(f"✅ Built vector index with {len(self.vector_store.keywords)} keywords")
            
            # Save the vector store
            logger.info(f"💾 Saving vector store to {Config.VECTOR_STORE_PATH}...")
            self.vector_store.save_store()
            logger.info("✅ Vector store saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            return False
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step
        workflow.add_node("start", self._start_node)
        workflow.add_node("normalize_keywords", self._normalize_keywords_node)
        workflow.add_node("inspect_schema", self._inspect_schema_node)
        workflow.add_node("generate_sql", self._generate_sql_node)
        workflow.add_node("execute_query", self._execute_query_node)
        workflow.add_node("format_results", self._format_results_node)
        workflow.add_node("generate_chart", self._generate_chart_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow edges
        workflow.set_entry_point("start")
        
        # Use conditional edges for the entire flow to handle errors
        workflow.add_conditional_edges(
            "start",
            self._should_continue,
            {
                "continue": "normalize_keywords",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "normalize_keywords",
            self._should_continue,
            {
                "continue": "inspect_schema",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "inspect_schema",
            self._should_continue,
            {
                "continue": "generate_sql",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_continue,
            {
                "continue": "execute_query",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_query",
            self._should_continue,
            {
                "continue": "format_results",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "format_results",
            self._should_continue,
            {
                "continue": "generate_chart",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_chart",
            self._should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        # Error handling
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _build_basic_workflow(self) -> StateGraph:
        """Build the basic workflow (without chart generation)"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step (excluding chart generation)
        workflow.add_node("start", self._start_node)
        workflow.add_node("normalize_keywords", self._normalize_keywords_node)
        workflow.add_node("inspect_schema", self._inspect_schema_node)
        workflow.add_node("generate_sql", self._generate_sql_node)
        workflow.add_node("execute_query", self._execute_query_node)
        workflow.add_node("format_results", self._format_results_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow edges
        workflow.set_entry_point("start")
        
        # Use conditional edges for the entire flow to handle errors
        workflow.add_conditional_edges(
            "start",
            self._should_continue,
            {
                "continue": "normalize_keywords",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "normalize_keywords",
            self._should_continue,
            {
                "continue": "inspect_schema",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "inspect_schema",
            self._should_continue,
            {
                "continue": "generate_sql",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_sql",
            self._should_continue,
            {
                "continue": "execute_query",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_query",
            self._should_continue,
            {
                "continue": "format_results",
                "error": "handle_error"
            }
        )
        
        # End after format_results (no chart generation)
        workflow.add_conditional_edges(
            "format_results",
            self._should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        # Error handling
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _build_chart_workflow(self) -> StateGraph:
        """Build the chart generation workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for chart generation only
        workflow.add_node("start", self._start_node)
        workflow.add_node("generate_chart", self._generate_chart_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow edges
        workflow.set_entry_point("start")
        
        # Go directly to chart generation
        workflow.add_conditional_edges(
            "start",
            self._should_continue,
            {
                "continue": "generate_chart",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_chart",
            self._should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        # Error handling
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _start_node(self, state: AgentState) -> AgentState:
        """Initialize the workflow"""
        state.update_step("start")
        state.start_timing()
        logger.info(f"Starting workflow for query: {state.user_query}")
        return state
    
    def _normalize_keywords_node(self, state: AgentState) -> AgentState:
        """Normalize keywords in the user query"""
        step_start_time = time.time()
        try:
            state.update_step("normalize_keywords")
            logger.info("Normalizing keywords...")
            
            # Execute keyword normalization
            result = self.tool_executor.invoke({
                "tool": "keyword_normalizer",
                "tool_input": {
                    "query": state.user_query,
                    "force_rebuild": False
                }
            })
            
            if result.get('output', {}).get('success'):
                data = result['output']['data']
                state.normalized_query = data['normalized_query']
                state.replacements = data.get('replacements', [])
                
                # Add execution time to result
                result['output']['execution_time'] = time.time() - step_start_time
                state.add_tool_call("keyword_normalizer", state.user_query, 
                                  ToolResult(**result['output']))
                state.add_step_timing("normalize_keywords", result['output']['execution_time'])
                logger.info(f"Keywords normalized: {len(data.get('replacements', []))} replacements made in {result['output']['execution_time']:.2f}s")
            else:
                state.add_error(f"Keyword normalization failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in keyword normalization: {e}")
            state.add_error(str(e))
            state.increment_retry()
        
        return state
    
    def _inspect_schema_node(self, state: AgentState) -> AgentState:
        """Inspect database schema"""
        step_start_time = time.time()
        try:
            state.update_step("inspect_schema")
            logger.info("Inspecting database schema...")
            
            # Execute schema inspection
            result = self.tool_executor.invoke({
                "tool": "schema_inspector",
                "tool_input": {
                    "table_name": None  # Get all schemas
                }
            })
            
            if result.get('output', {}).get('success'):
                state.database_schemas = result['output']['data']
                
                # Add execution time to result
                result['output']['execution_time'] = time.time() - step_start_time
                state.add_tool_call("schema_inspector", None, ToolResult(**result['output']))
                state.add_step_timing("inspect_schema", result['output']['execution_time'])
                logger.info(f"Schema inspection complete: {len(result['output']['data'])} tables in {result['output']['execution_time']:.2f}s")
            else:
                state.add_error(f"Schema inspection failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in schema inspection: {e}")
            state.add_error(str(e))
            state.increment_retry()
        
        return state
    
    def _generate_sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL query from natural language"""
        step_start_time = time.time()
        try:
            state.update_step("generate_sql")
            logger.info("Generating SQL query...")
            
            # Execute SQL generation
            result = self.tool_executor.invoke({
                "tool": "sql_generator",
                "tool_input": {
                    "user_query": state.user_query,
                    "database_schemas": state.database_schemas,
                    "normalized_query": state.normalized_query,
                    "replacements": state.replacements
                }
            })
            
            if result.get('output', {}).get('success'):
                data = result['output']['data']
                state.generated_sql = data['sql_query']
                
                # Add execution time to result
                result['output']['execution_time'] = time.time() - step_start_time
                state.add_tool_call("sql_generator", state.user_query, ToolResult(**result['output']))
                state.add_step_timing("generate_sql", result['output']['execution_time'])
                logger.info(f"SQL generated successfully: {data['sql_query'][:100]}... in {result['output']['execution_time']:.2f}s")
            else:
                state.add_error(f"SQL generation failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in SQL generation: {e}")
            state.add_error(str(e))
            state.increment_retry()
        
        return state
    
    def _execute_query_node(self, state: AgentState) -> AgentState:
        """Execute the generated SQL query"""
        step_start_time = time.time()
        try:
            state.update_step("execute_query")
            logger.info("Executing SQL query...")
            
            # Execute database query
            result = self.tool_executor.invoke({
                "tool": "db_query",
                "tool_input": {
                    "sql_query": state.generated_sql
                }
            })
            
            if result.get('output', {}).get('success'):
                state.sql_execution_result = result['output']['data']
                
                # Add execution time to result
                result['output']['execution_time'] = time.time() - step_start_time
                state.add_tool_call("db_query", state.generated_sql, ToolResult(**result['output']))
                state.add_step_timing("execute_query", result['output']['execution_time'])
                logger.info(f"Query executed successfully: {result['output']['metadata'].get('rows_returned', 0)} rows returned in {result['output']['execution_time']:.2f}s")
            else:
                state.add_error(f"Query execution failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in query execution: {e}")
            state.add_error(str(e))
            state.increment_retry()
        
        return state
    
    def _format_results_node(self, state: AgentState) -> AgentState:
        """Format results into natural language response"""
        step_start_time = time.time()
        try:
            state.update_step("format_results")
            logger.info("Formatting results...")
            
            # Execute result formatting
            result = self.tool_executor.invoke({
                "tool": "result_formatter",
                "tool_input": {
                    "user_query": state.user_query,
                    "sql_query": state.generated_sql,
                    "query_results": state.sql_execution_result,
                    "database_schemas": state.database_schemas
                }
            })
            
            if result.get('output', {}).get('success'):
                data = result['output']['data']
                state.final_response = data['natural_response']
                
                # Add execution time to result
                result['output']['execution_time'] = time.time() - step_start_time
                state.add_tool_call("result_formatter", "format_results", ToolResult(**result['output']))
                state.add_step_timing("format_results", result['output']['execution_time'])
                logger.info(f"Results formatted successfully in {result['output']['execution_time']:.2f}s")
            else:
                state.add_error(f"Result formatting failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in result formatting: {e}")
            state.add_error(str(e))
            state.increment_retry()
        
        return state
    
    def _generate_chart_node(self, state: AgentState) -> AgentState:
        """Generate visualization code for the query results"""
        step_start_time = time.time()
        try:
            state.update_step("generate_chart")
            logger.info("Generating chart...")
            
            # Execute chart generation
            result = self.tool_executor.invoke({
                "tool": "chart_generator",
                "tool_input": {
                    "user_query": state.user_query,
                    "query_results": state.sql_execution_result,
                    "database_schemas": state.database_schemas
                }
            })
            
            if result.get('output', {}).get('success'):
                data = result['output']['data']
                state.visualization_code = data['visualization_code']
                
                # Add execution time to result
                result['output']['execution_time'] = time.time() - step_start_time
                state.add_tool_call("chart_generator", "generate_chart", ToolResult(**result['output']))
                state.add_step_timing("generate_chart", result['output']['execution_time'])
                logger.info(f"Chart generated successfully in {result['output']['execution_time']:.2f}s")
            else:
                state.add_error(f"Chart generation failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in chart generation: {e}")
            state.add_error(str(e))
            state.increment_retry()
        
        return state
    
    def _handle_error_node(self, state: AgentState) -> AgentState:
        """Handle errors and decide on next steps"""
        state.update_step("error_handling")
        logger.warning(f"Error handling step: {len(state.errors)} errors encountered")
        
        # If we have too many retries, stop
        if state.retry_count >= state.max_retries:
            state.should_continue = False
            logger.error("Maximum retries reached, stopping workflow")
        else:
            # Could implement retry logic here
            logger.info(f"Retry {state.retry_count + 1} of {state.max_retries}")
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the workflow should continue or handle errors"""
        if state.errors and state.retry_count >= state.max_retries:
            return "error"
        elif state.errors:
            return "error"
        else:
            return "continue"
    
    def process_query(self, user_query: str) -> AgentState:
        """Process a user query through the complete workflow"""
        try:
            # Initialize state
            initial_state = AgentState(user_query=user_query)
            
            # Execute the workflow
            workflow_result = self.workflow.invoke(initial_state)
            
            # LangGraph returns a dictionary, convert it back to AgentState
            if isinstance(workflow_result, dict):
                final_state = AgentState(user_query=user_query)
                # Copy all attributes from the dictionary to the state object
                for key, value in workflow_result.items():
                    if hasattr(final_state, key):
                        setattr(final_state, key, value)
                
                # End timing and log summary
                final_state.end_timing()
                timing_summary = final_state.get_timing_summary()
                logger.info(f"Workflow completed successfully in {final_state.total_execution_time:.2f}s")
                logger.info(f"Step timings: {timing_summary['step_timings']}")
                
                return final_state
            else:
                # End timing for non-dict results
                workflow_result.end_timing()
                timing_summary = workflow_result.get_timing_summary()
                logger.info(f"Workflow completed successfully in {workflow_result.total_execution_time:.2f}s")
                logger.info(f"Step timings: {timing_summary['step_timings']}")
                return workflow_result
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            # Return error state
            error_state = AgentState(user_query=user_query)
            error_state.add_error(str(e))
            error_state.should_continue = False
            error_state.end_timing()
            return error_state
    
    def process_query_basic(self, user_query: str) -> AgentState:
        """Process a user query through basic workflow (without chart generation)"""
        try:
            # Initialize state
            initial_state = AgentState(user_query=user_query)
            
            # Execute the basic workflow (without chart generation)
            workflow_result = self.basic_workflow.invoke(initial_state)
            
            # LangGraph returns a dictionary, convert it back to AgentState
            if isinstance(workflow_result, dict):
                final_state = AgentState(user_query=user_query)
                # Copy all attributes from the dictionary to the state object
                for key, value in workflow_result.items():
                    if hasattr(final_state, key):
                        setattr(final_state, key, value)
                
                # End timing and log summary
                final_state.end_timing()
                timing_summary = final_state.get_timing_summary()
                logger.info(f"Basic workflow completed successfully in {final_state.total_execution_time:.2f}s")
                logger.info(f"Step timings: {timing_summary['step_timings']}")
                
                return final_state
            else:
                # End timing for non-dict results
                workflow_result.end_timing()
                timing_summary = workflow_result.get_timing_summary()
                logger.info(f"Basic workflow completed successfully in {workflow_result.total_execution_time:.2f}s")
                logger.info(f"Step timings: {timing_summary['step_timings']}")
                return workflow_result
            
        except Exception as e:
            logger.error(f"Error in basic workflow execution: {e}")
            # Return error state
            error_state = AgentState(user_query=user_query)
            error_state.add_error(str(e))
            error_state.should_continue = False
            error_state.end_timing()
            return error_state
    
    def generate_chart_for_data(self, user_query: str, sql_execution_result, database_schemas) -> AgentState:
        """Generate chart for existing data"""
        try:
            # Initialize state with existing data
            initial_state = AgentState(
                user_query=user_query,
                sql_execution_result=sql_execution_result,
                database_schemas=database_schemas
            )
            
            # Execute only the chart generation workflow
            workflow_result = self.chart_workflow.invoke(initial_state)
            
            # LangGraph returns a dictionary, convert it back to AgentState
            if isinstance(workflow_result, dict):
                final_state = AgentState(user_query=user_query)
                # Copy all attributes from the dictionary to the state object
                for key, value in workflow_result.items():
                    if hasattr(final_state, key):
                        setattr(final_state, key, value)
                
                # End timing and log summary
                final_state.end_timing()
                timing_summary = final_state.get_timing_summary()
                logger.info(f"Chart generation completed successfully in {final_state.total_execution_time:.2f}s")
                logger.info(f"Step timings: {timing_summary['step_timings']}")
                
                return final_state
            else:
                # End timing for non-dict results
                workflow_result.end_timing()
                timing_summary = workflow_result.get_timing_summary()
                logger.info(f"Chart generation completed successfully in {workflow_result.total_execution_time:.2f}s")
                logger.info(f"Step timings: {timing_summary['step_timings']}")
                return workflow_result
            
        except Exception as e:
            logger.error(f"Error in chart generation: {e}")
            # Return error state
            error_state = AgentState(user_query=user_query)
            error_state.add_error(str(e))
            error_state.should_continue = False
            error_state.end_timing()
            return error_state
    
    def close(self):
        """Clean up resources"""
        try:
            self.db_manager.close()
            logger.info("Agent planner resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
