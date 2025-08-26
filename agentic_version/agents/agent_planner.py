from typing import Dict, List, Any, Optional, Tuple
from langgraph.graph import StateGraph, END
from langchain.tools import BaseTool
from config.state import AgentState, ToolResult
from tools.tool_definitions import create_tools
from tools.database_manager import DatabaseManager
from tools.vector_store import FAISSVectorStore
from tools.llm_manager import LLMManager
import logging

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
        
        # Create tools
        self.tools = create_tools(self.db_manager, self.vector_store, self.llm_manager)
        self.tool_executor = ToolExecutor(self.tools)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Agent planner initialized successfully")
    
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
        logger.info(f"Starting workflow for query: {state.user_query}")
        return state
    
    def _normalize_keywords_node(self, state: AgentState) -> AgentState:
        """Normalize keywords in the user query"""
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
                state.add_tool_call("keyword_normalizer", state.user_query, 
                                  ToolResult(**result['output']))
                logger.info(f"Keywords normalized: {len(data.get('replacements', []))} replacements made")
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
                state.add_tool_call("schema_inspector", None, ToolResult(**result['output']))
                logger.info(f"Schema inspection complete: {len(result['output']['data'])} tables")
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
        try:
            state.update_step("generate_sql")
            logger.info("Generating SQL query...")
            
            # Execute SQL generation
            result = self.tool_executor.invoke({
                "tool": "sql_generator",
                "tool_input": {
                    "user_query": state.user_query,
                    "database_schemas": state.database_schemas,
                    "normalized_query": state.normalized_query
                }
            })
            
            if result.get('output', {}).get('success'):
                data = result['output']['data']
                state.generated_sql = data['sql_query']
                state.add_tool_call("sql_generator", state.user_query, ToolResult(**result['output']))
                logger.info(f"SQL generated successfully: {data['sql_query'][:100]}...")
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
                state.add_tool_call("db_query", state.generated_sql, ToolResult(**result['output']))
                logger.info(f"Query executed successfully: {result['output']['metadata'].get('rows_returned', 0)} rows returned")
            else:
                state.add_error(f"Query execution failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in query execution: {e}")
            state.add_error(str(e))
            state.increment_retry()
        
        return state
    
    def _format_results_node(self, state: AgentState) -> AgentState:
        """Format results into natural language and visualization"""
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
                state.visualization_code = data['visualization_code']
                state.add_tool_call("result_formatter", "format_results", ToolResult(**result['output']))
                logger.info("Results formatted successfully")
            else:
                state.add_error(f"Result formatting failed: {result.get('output', {}).get('error')}")
                state.increment_retry()
            
        except Exception as e:
            logger.error(f"Error in result formatting: {e}")
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
                logger.info("Workflow completed successfully")
                return final_state
            else:
                logger.info("Workflow completed successfully")
                return workflow_result
            
        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            # Return error state
            error_state = AgentState(user_query=user_query)
            error_state.add_error(str(e))
            error_state.should_continue = False
            return error_state
    
    def close(self):
        """Clean up resources"""
        try:
            self.db_manager.close()
            logger.info("Agent planner resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
