from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.state import ToolResult
from tools.database_manager import DatabaseManager
from tools.vector_store import FAISSVectorStore
from tools.llm_manager import LLMManager
import logging

logger = logging.getLogger(__name__)

# Input/Output schemas for tools
class KeywordNormalizerInput(BaseModel):
    query: str = Field(description="Natural language query to normalize")
    force_rebuild: bool = Field(default=False, description="Force rebuild of vector store")

class SchemaInspectorInput(BaseModel):
    table_name: Optional[str] = Field(default=None, description="Specific table to inspect, or None for all tables")

class SQLGeneratorInput(BaseModel):
    user_query: str = Field(description="Natural language query")
    database_schemas: Dict[str, Any] = Field(description="Database schema information")
    normalized_query: Optional[str] = Field(default=None, description="Normalized query with corrected keywords")
    replacements: Optional[List[Dict[str, Any]]] = Field(default=None, description="Keyword replacements made during normalization")

class DBQueryInput(BaseModel):
    sql_query: str = Field(description="SQL query to execute")

class ResultFormatterInput(BaseModel):
    user_query: str = Field(description="Original user query")
    sql_query: str = Field(description="Executed SQL query")
    query_results: Any = Field(description="Results from SQL execution")
    database_schemas: Dict[str, Any] = Field(description="Database schema information")

class ChartGeneratorInput(BaseModel):
    user_query: str = Field(description="Original user query")
    query_results: Any = Field(description="Results from SQL execution")
    database_schemas: Dict[str, Any] = Field(description="Database schema information")

# Tool implementations
class KeywordNormalizerTool(BaseTool):
    name: str = "keyword_normalizer"
    description: str = "Normalize keywords in natural language queries using vector similarity search"
    args_schema: type[KeywordNormalizerInput] = KeywordNormalizerInput
    
    def __init__(self, vector_store: FAISSVectorStore):
        super().__init__()
        self._vector_store = vector_store
    
    def _run(self, query: str, force_rebuild: bool = False) -> ToolResult:
        """Normalize keywords in the query"""
        try:
            if force_rebuild:
                self._vector_store.reset_store()
                # Rebuild would need database access - simplified for now
                logger.info("Vector store reset requested")
            
            # Normalize the query
            result = self._vector_store.normalize_query_keywords(query)
            
            if result['success']:
                return ToolResult(
                    success=True,
                    data=result,
                    metadata={
                        'replacements_made': result['replacements'],
                        'original_query': query,
                        'normalized_query': result['normalized_query']
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    data=result,
                    error=result.get('error', 'Unknown error in keyword normalization')
                )
                
        except Exception as e:
            logger.error(f"Error in keyword normalization: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

class SchemaInspectorTool(BaseTool):
    name: str = "schema_inspector"
    description: str = "Retrieve database schema and metadata information"
    args_schema: type[SchemaInspectorInput] = SchemaInspectorInput
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db_manager = db_manager
    
    def _run(self, table_name: Optional[str] = None) -> ToolResult:
        """Get database schema information"""
        try:
            if table_name:
                schema = self._db_manager.get_table_schema(table_name)
                if schema:
                    return ToolResult(
                        success=True,
                        data={table_name: schema},
                        metadata={'tables_inspected': 1}
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Table {table_name} not found or error accessing schema"
                    )
            else:
                schemas = self._db_manager.get_all_schemas()
                return ToolResult(
                    success=True,
                    data=schemas,
                    metadata={'tables_inspected': len(schemas)}
                )
                
        except Exception as e:
            logger.error(f"Error in schema inspection: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

class SQLGeneratorTool(BaseTool):
    name: str = "sql_generator"
    description: str = "Generate SQL queries from natural language using LLM"
    args_schema: type[SQLGeneratorInput] = SQLGeneratorInput
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__()
        self._llm_manager = llm_manager
    
    def _run(self, user_query: str, database_schemas: Dict[str, Any], 
             normalized_query: Optional[str] = None, replacements: Optional[List[Dict[str, Any]]] = None) -> ToolResult:
        """Generate SQL query from natural language"""
        try:
            result = self._llm_manager.generate_sql_query(
                user_query, database_schemas, normalized_query, replacements
            )
            
            if result['success']:
                # Validate the generated SQL
                validation = self._llm_manager.validate_sql_query(result['sql_query'])
                
                return ToolResult(
                    success=True,
                    data={
                        'sql_query': result['sql_query'],
                        'validation': validation,
                        'model_used': result['model_used']
                    },
                    metadata={
                        'tokens_used': result.get('tokens_used'),
                        'validation_passed': validation
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    data=result,
                    error=result.get('error', 'Unknown error in SQL generation')
                )
                
        except Exception as e:
            logger.error(f"Error in SQL generation: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

class DBQueryTool(BaseTool):
    name: str = "db_query"
    description: str = "Execute SQL queries against the PostgreSQL database"
    args_schema: type[DBQueryInput] = DBQueryInput
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db_manager = db_manager
    
    def _run(self, sql_query: str) -> ToolResult:
        """Execute SQL query"""
        try:
            # Execute the query
            results = self._db_manager.execute_query(sql_query)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    'rows_returned': len(results) if hasattr(results, '__len__') else 0,
                    'query_type': 'SELECT' if 'SELECT' in sql_query.upper() else 'OTHER'
                }
            )
                
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

class ResultFormatterTool(BaseTool):
    name: str = "result_formatter"
    description: str = "Generate natural language explanations for query results"
    args_schema: type[ResultFormatterInput] = ResultFormatterInput
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__()
        self._llm_manager = llm_manager
    
    def _run(self, user_query: str, sql_query: str, query_results: Any, 
             database_schemas: Dict[str, Any]) -> ToolResult:
        """Format results into natural language response"""
        try:
            # Generate natural language response only
            natural_response = self._llm_manager.generate_natural_response(
                user_query, sql_query, query_results, database_schemas
            )
            
            if natural_response['success']:
                return ToolResult(
                    success=True,
                    data={
                        'natural_response': natural_response['natural_response'],
                        'model_used': natural_response['model_used']
                    },
                    metadata={
                        'natural_response_tokens': natural_response.get('tokens_used')
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Natural response generation failed: {natural_response.get('error')}"
                )
                
        except Exception as e:
            logger.error(f"Error in result formatting: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

class ChartGeneratorTool(BaseTool):
    name: str = "chart_generator"
    description: str = "Generate visualization code for query results"
    args_schema: type[ChartGeneratorInput] = ChartGeneratorInput
    
    def __init__(self, llm_manager: LLMManager):
        super().__init__()
        self._llm_manager = llm_manager
    
    def _run(self, user_query: str, query_results: Any, 
             database_schemas: Dict[str, Any]) -> ToolResult:
        """Generate visualization code for the query results"""
        try:
            # Generate visualization code
            visualization = self._llm_manager.generate_visualization_code(
                user_query, query_results, database_schemas
            )
            
            if visualization['success']:
                return ToolResult(
                    success=True,
                    data={
                        'visualization_code': visualization['visualization_code'],
                        'model_used': visualization['model_used']
                    },
                    metadata={
                        'visualization_tokens': visualization.get('tokens_used')
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Visualization generation failed: {visualization.get('error')}"
                )
                
        except Exception as e:
            logger.error(f"Error in chart generation: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

def create_tools(db_manager: DatabaseManager, vector_store: FAISSVectorStore, 
                llm_manager: LLMManager) -> List[BaseTool]:
    """Create all tools for the agent system"""
    return [
        KeywordNormalizerTool(vector_store),
        SchemaInspectorTool(db_manager),
        SQLGeneratorTool(llm_manager),
        DBQueryTool(db_manager),
        ResultFormatterTool(llm_manager),
        ChartGeneratorTool(llm_manager)
    ]
