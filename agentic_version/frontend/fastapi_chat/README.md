# FastAPI Chat Interface

A modern web-based chat interface for the Agentic AI System using FastAPI and Jinja2 templates.

## Features

- **Real-time Chat Interface**: Clean, modern chat UI with typing indicators
- **Natural Language Processing**: Same AI processing as the Streamlit app
- **Interactive Visualizations**: Automatic chart generation and display
- **Example Queries**: Quick access to common queries
- **Chat History**: Export and clear chat functionality
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python run.py
```

Or directly with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. Open your browser and go to `http://localhost:8000`
2. Type your natural language query in the input field
3. Click "Send" or press Enter
4. View the AI response and any generated charts
5. Use example queries from the sidebar for quick testing

## API Endpoints

- `GET /` - Main chat interface
- `POST /api/chat` - Process chat messages
- `GET /api/health` - Health check
- `GET /api/examples` - Get example queries

## Architecture

- **Backend**: FastAPI with async support
- **Frontend**: HTML5 + Bootstrap 5 + JavaScript
- **Templates**: Jinja2 for server-side rendering
- **Styling**: Custom CSS with responsive design
- **Charts**: Plotly.js for interactive visualizations

## File Structure

```
fastapi_chat/
├── main.py              # FastAPI application
├── run.py               # Launcher script
├── requirements.txt     # Dependencies
├── templates/
│   └── chat.html       # Main chat template
└── static/
    ├── css/
    │   └── chat.css    # Chat styling
    └── js/
        └── chat.js     # Chat functionality
```

## Integration

This chat interface uses the same agentic system as the Streamlit app but provides a different user experience. It's completely separate and won't interfere with your existing Streamlit testing environment.

The interface connects to the same backend components:
- `AgentPlanner` for query processing
- Same visualization generation
- Same natural language responses
- Same error handling


