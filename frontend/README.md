# 🤖 Agentic AI Query System - Frontend

A beautiful Streamlit web interface for the Agentic AI System that transforms natural language queries into SQL and provides intelligent insights from your database.

## 🚀 Features

- **Natural Language Interface**: Simply type your questions in plain English
- **Real-time Processing**: Watch the AI process your query step by step
- **Interactive Results**: View generated SQL, data tables, and visualizations
- **Tool Execution Details**: See exactly how each AI tool processes your query
- **Example Queries**: Get started quickly with pre-built examples
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🛠️ Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the frontend**:
   ```bash
   python run.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and go to: `http://localhost:8501`

## 🎯 How to Use

1. **Enter a Query**: Type your natural language question in the text area
2. **Choose Examples**: Use the sidebar to select from pre-built example queries
3. **Process Query**: Click the "Process Query" button to start the AI workflow
4. **View Results**: Explore the generated SQL, data results, and insights
5. **Explore Details**: Expand each tool execution to see the detailed process

## 🔧 Example Queries

- "Show me data from Riyadh Office"
- "What are the top legal entity types by contributor count?"
- "How many contributors are there in the technology sector?"
- "Show the breakdown of contributors by occupation group"
- "Which cities have the highest number of private entities?"
- "Compare construction and commerce sectors across cities"

## 🔗 Backend Integration

This frontend seamlessly connects to your agentic AI backend system:
- **Agent Planner**: Orchestrates the multi-step workflow
- **Vector Store**: Provides intelligent keyword normalization
- **Database Manager**: Handles PostgreSQL connections and queries
- **LLM Manager**: Generates SQL and natural language responses
- **Tool System**: Executes specialized AI tools for each step

## 🎉 Enjoy!

Your agentic AI system now has a beautiful, user-friendly interface that makes it easy to explore your data with natural language queries!
