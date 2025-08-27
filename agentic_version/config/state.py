from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ToolName(str, Enum):
    """Available tools in the system"""
    KEYWORD_NORMALIZER = "keyword_normalizer"
    SCHEMA_INSPECTOR = "schema_inspector"
    SQL_GENERATOR = "sql_generator"
    DB_QUERY = "db_query"
    RESULT_FORMATTER = "result_formatter"

class ToolResult(BaseModel):
    """Result from a tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentState(BaseModel):
    """State maintained throughout the agent workflow"""
    
    # User input and context
    user_query: str = Field(description="Original user query")
    current_step: str = Field(default="start", description="Current step in the workflow")
    
    # Tool execution history
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="History of tool calls")
    tool_results: Dict[str, ToolResult] = Field(default_factory=dict, description="Results from tool executions")
    
    # Database and schema information
    database_schemas: Optional[Dict[str, Any]] = Field(default=None, description="Database schema information")
    normalized_query: Optional[str] = Field(default=None, description="Query with normalized keywords")
    replacements: Optional[List[Dict[str, Any]]] = Field(default=None, description="Keyword replacements made during normalization")
    
    # SQL generation and execution
    generated_sql: Optional[str] = Field(default=None, description="Generated SQL query")
    sql_execution_result: Optional[Any] = Field(default=None, description="Result from SQL execution")
    
    # Final output
    final_response: Optional[str] = Field(default=None, description="Final natural language response")
    visualization_code: Optional[str] = Field(default=None, description="Generated visualization code")
    
    # Error handling and retry logic
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    
    # Workflow control
    should_continue: bool = Field(default=True, description="Whether the workflow should continue")
    next_tool: Optional[ToolName] = Field(default=None, description="Next tool to execute")
    
    def add_tool_call(self, tool_name: str, input_data: Any, result: ToolResult):
        """Add a tool call to the history"""
        self.tool_calls.append({
            "tool": tool_name,
            "input": input_data,
            "result": result,
            "step": self.current_step
        })
        self.tool_results[tool_name] = result
    
    def add_error(self, error: str):
        """Add an error to the error list"""
        self.errors.append(error)
    
    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.should_continue = False
    
    def is_complete(self) -> bool:
        """Check if the workflow is complete"""
        return (
            self.final_response is not None and 
            self.sql_execution_result is not None
        )
    
    def is_successful(self) -> bool:
        """Check if the workflow completed successfully without critical errors"""
        return (
            self.is_complete() and 
            len(self.errors) == 0 and
            self.retry_count < self.max_retries
        )
    
    def get_last_tool_result(self, tool_name: str) -> Optional[ToolResult]:
        """Get the last result from a specific tool"""
        return self.tool_results.get(tool_name)
    
    def update_step(self, step: str):
        """Update the current step"""
        self.current_step = step
