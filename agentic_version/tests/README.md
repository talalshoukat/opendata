# Tests for Agentic AI System

This directory contains all tests for the agentic AI system.

## Test Files

### `test_basic_functionality.py`
Tests the core functionality of the system:
- Module imports
- State management
- Vector store operations
- Tool definitions
- Workflow logic

### `test_workflow.py`
Tests the complete workflow with mocked LLM responses:
- Complete end-to-end workflow
- Vector store with real database data
- Tool execution
- State management throughout the workflow

### `test_api.py`
Tests OpenAI API integration:
- API connection
- LLM manager functionality
- SQL generation
- Response formatting

## Running Tests

### Run all tests:
```bash
python run_tests.py
```

### Run specific test:
```bash
python run_tests.py basic      # Run basic functionality tests
python run_tests.py workflow   # Run workflow tests
python run_tests.py api        # Run API tests
```

### Run individual test files:
```bash
python tests/test_basic_functionality.py
python tests/test_workflow.py
python tests/test_api.py
```

## Test Requirements

- Database connection (PostgreSQL)
- OpenAI API key (for API tests)
- All dependencies installed

## Test Coverage

The tests cover:
- ✅ Core system functionality
- ✅ Database connectivity
- ✅ Vector store operations
- ✅ Tool definitions and execution
- ✅ Workflow state management
- ✅ Error handling
- ✅ API integration (when available)

## Notes

- API tests require a valid OpenAI API key
- Workflow tests use mocked LLM responses to avoid API costs
- Database tests require a running PostgreSQL instance
- Vector store tests use real database data when available
