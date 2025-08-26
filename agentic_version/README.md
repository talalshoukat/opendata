# Agentic AI System for Natural Language to SQL Queries

A robust, multi-step agent system using **LangGraph** for intelligent query processing. This system demonstrates modern agentic AI architecture with modular tools, intelligent workflow planning, and robust error handling.

## 🚀 Key Features

- **Modular Tool Architecture**: Each capability is implemented as a separate, reusable tool
- **Intelligent Workflow Planning**: Uses LangGraph for stateful, multi-step execution
- **Robust Error Handling**: Built-in retry mechanisms and graceful failure handling
- **Vector-Based Keyword Normalization**: Handles typos and synonyms using FAISS
- **LLM-Powered SQL Generation**: Converts natural language to SQL using OpenAI
- **Natural Language Results**: Explains query results in plain English
- **Automated Visualization**: Generates Python code for data visualization

## 🏗️ Architecture Overview

The system follows a **ReAct (Reasoning + Acting)** pattern with the following workflow:

```
User Query → Keyword Normalization → Schema Inspection → SQL Generation → Query Execution → Result Formatting
```

### Core Components

1. **Agent Planner** (`agents/agent_planner.py`): Orchestrates the workflow using LangGraph
2. **Tools** (`tools/`): Modular components for specific tasks
3. **State Management** (`config/state.py`): Pydantic-based state tracking
4. **Configuration** (`config/config.py`): Centralized configuration management

### Available Tools

- **`keyword_normalizer`**: Normalizes keywords using vector similarity search
- **`schema_inspector`**: Retrieves database schema and metadata
- **`sql_generator`**: Generates SQL from natural language using LLM
- **`db_query`**: Executes SQL queries against PostgreSQL
- **`result_formatter`**: Generates natural language explanations and visualization code

## 📋 Requirements

### Python Dependencies
```bash
# Core agent framework
langgraph==0.2.0
langchain==0.2.0
langchain-openai==0.1.0
langchain-community==0.2.0

# Database and data processing
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
pandas==2.0.3
numpy==1.24.3

# AI and embeddings
openai==1.3.7
faiss-cpu==1.7.4
scikit-learn==1.3.0

# Additional utilities
python-dotenv==1.0.0
pydantic==2.5.0
```

### Environment Variables
Create a `.env` file in the project root:

```env
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=open_data
DB_USER=your_username
DB_PASSWORD=your_password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo

# Optional Configuration
VECTOR_STORE_PATH=./vector_store
MAX_RETRIES=3
TEMPERATURE=0.1
SIMILARITY_THRESHOLD=0.7
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Using conda (recommended)
conda create -n open_data_agentic python=3.11
conda activate open_data_agentic

# Install packages
pip install -r requirements.txt

# Install FAISS and scikit-learn via conda
conda install -c conda-forge faiss-cpu scikit-learn
```

### 2. Set Up Environment
```bash
# Copy and configure environment variables
cp env_example.txt .env
# Edit .env with your actual values
```

### 3. Run the System
```bash
# Run demo queries
python main.py demo

# Run in interactive mode
python main.py interactive

# Run demo (default)
python main.py
```

## 🔧 Usage Examples

### Example 1: Basic Query Processing
```python
from agents.agent_planner import AgentPlanner

# Initialize the agent
agent = AgentPlanner()

# Process a query
result = agent.process_query("Show me the distribution of contributors by economic activity")

# Check results
if result.is_complete():
    print(f"SQL Generated: {result.generated_sql}")
    print(f"Response: {result.final_response}")
    print(f"Visualization Code: {result.visualization_code}")
```

### Example 2: Custom Tool Usage
```python
from tools.tool_definitions import create_tools
from tools.database_manager import DatabaseManager
from tools.vector_store import FAISSVectorStore
from tools.llm_manager import LLMManager

# Create individual tools
db_manager = DatabaseManager()
vector_store = FAISSVectorStore()
llm_manager = LLMManager()

# Use specific tools
tools = create_tools(db_manager, vector_store, llm_manager)
keyword_tool = tools[0]  # KeywordNormalizerTool

# Normalize keywords
result = keyword_tool._run("Show me tech sector data")
print(f"Normalized: {result.data['normalized_query']}")
```

## 🏗️ Extending the System

### Adding New Tools

1. **Create Tool Class**: Inherit from `BaseTool` and implement `_run` method
2. **Define Input Schema**: Create Pydantic model for tool inputs
3. **Add to Tool Factory**: Update `create_tools()` function in `tools/tool_definitions.py`
4. **Update Workflow**: Add new nodes to the agent planner if needed

Example:
```python
class NewToolInput(BaseModel):
    input_data: str = Field(description="Input for the new tool")

class NewTool(BaseTool):
    name = "new_tool"
    description = "Description of what this tool does"
    args_schema = NewToolInput
    
    def _run(self, input_data: str) -> ToolResult:
        # Tool implementation
        return ToolResult(success=True, data="result")
```

### Adding New Workflow Steps

1. **Create Node Method**: Add new method to `AgentPlanner` class
2. **Update Graph**: Add node and edges in `_build_workflow()`
3. **Update State**: Add new fields to `AgentState` if needed

## 🔍 System Monitoring

### Logging
The system provides comprehensive logging:
- **File Logging**: `agentic_system.log`
- **Console Output**: Real-time workflow status
- **Tool Execution Tracking**: Detailed tool call history

### State Inspection
```python
# Get current workflow state
state = agent.process_query("your query")

# Inspect tool execution history
for tool_call in state.tool_calls:
    print(f"Tool: {tool_call['tool']}")
    print(f"Step: {tool_call['step']}")
    print(f"Success: {tool_call['result'].success}")

# Check for errors
if state.errors:
    print(f"Errors: {state.errors}")
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_agent_planner.py

# Run with coverage
python -m pytest --cov=agents --cov=tools tests/
```

### Test Structure
```
tests/
├── test_agent_planner.py    # Agent workflow tests
├── test_tools.py            # Individual tool tests
├── test_config.py           # Configuration tests
└── test_integration.py      # End-to-end tests
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check database credentials in `.env`
   - Ensure PostgreSQL is running
   - Verify network connectivity

2. **OpenAI API Errors**
   - Verify API key is correct
   - Check API quota and billing
   - Ensure model name is valid

3. **Vector Store Issues**
   - Check FAISS installation
   - Verify scikit-learn version compatibility
   - Clear and rebuild vector store if needed

4. **Import Errors**
   - Ensure conda environment is activated
   - Check Python path and package installation
   - Verify `__init__.py` files exist

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py

# Or modify config/config.py
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### AgentPlanner Class
- `__init__()`: Initialize the agent system
- `process_query(query: str)`: Process a natural language query
- `close()`: Clean up resources

### Tool Classes
- `KeywordNormalizerTool`: Normalize query keywords
- `SchemaInspectorTool`: Inspect database schemas
- `SQLGeneratorTool`: Generate SQL from natural language
- `DBQueryTool`: Execute SQL queries
- `ResultFormatterTool`: Format and explain results

### State Management
- `AgentState`: Main state object for workflow tracking
- `ToolResult`: Result object for tool executions
- `ToolName`: Enumeration of available tools

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph**: For the powerful workflow orchestration framework
- **LangChain**: For the comprehensive LLM integration tools
- **OpenAI**: For the advanced language models
- **FAISS**: For efficient vector similarity search

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation
- Examine the example code

---

**Built with ❤️ using modern AI frameworks and best practices**
