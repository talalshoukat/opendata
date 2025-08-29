import openai
from typing import Dict, List, Any, Optional
import logging
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """Manager for OpenAI LLM interactions"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS
    
    def generate_sql_query(self, user_query: str, database_schemas: Dict[str, Any], 
                          normalized_query: str = None, replacements: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate SQL query from natural language using LLM"""
        try:
            schema_text = self._format_schema_for_llm(database_schemas)
            
            # Format replacement information if available
            replacement_text = ""
            if replacements:
                replacement_text = self._format_values_for_llm(replacements)
            
            # Prepare the prompt
            prompt = f"""
            You are a SQL expert. Generate a PostgreSQL query based on the user's request.
            
            User Request: {normalized_query or user_query}
            
            Database Schema:
            {schema_text}
            
            {replacement_text}
            
            Instructions:
            1. Generate a valid PostgreSQL query
            2. Use the exact table and column names from the schema
            3. Take data from 1 table only as they are aggregated table and cant be union or joined
            4. Use proper PostgreSQL syntax
            5. Include appropriate WHERE clauses based on the query intent
            6. Make sure no error in the query and not to use words like sql in query
            7. Use the relevant database values when appropriate for filtering
            8. When keyword replacements are provided, use the actual database values in your WHERE clauses
            9. If multiple similar matches are provided for a keyword, choose the most contextually appropriate one
            10. Return only the SQL query, no explanations
            
            SQL Query:
            """

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL developer. Generate only valid PostgreSQL SQL queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL query (remove markdown formatting if present)
            if sql_query.startswith('```sql'):
                sql_query = sql_query[7:]
            if sql_query.endswith('```'):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            return {
                'success': True,
                'sql_query': sql_query,
                'model_used': self.model,
                'tokens_used': response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return {
                'success': False,
                'error': str(e),
                'sql_query': None
            }
    
    def generate_natural_response(self, user_query: str, sql_query: str, 
                                query_results: Any, database_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """Generate natural language response explaining the query results"""
        try:
            # Prepare the prompt
            prompt = self._create_response_generation_prompt(user_query, sql_query, query_results, database_schemas)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst. Explain query results in clear, natural language. Keep responses concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            natural_response = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'natural_response': natural_response,
                'model_used': self.model,
                'tokens_used': response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            return {
                'success': False,
                'error': str(e),
                'natural_response': None
            }
    
    def generate_visualization_code(self, user_query: str, data: Any,
                                  database_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Python code for data visualization using OpenAI Code Interpreter"""
        try:
            # Convert data to DataFrame if needed
            import pandas as pd
            if hasattr(data, 'shape'):
                df = data
            else:
                df = pd.DataFrame(data)
            
            if df.empty:
                return {
                    'success': False,
                    'error': 'No data available for visualization',
                    'visualization_code': None
                }

            # Use OpenAI Code Interpreter for chart generation
            visualization_code = self._generate_code_interpreter_chart(user_query, df)
            
            # Clean up the code (remove markdown formatting if present)
            if visualization_code.startswith('```python'):
                visualization_code = visualization_code[9:]
            if visualization_code.endswith('```'):
                visualization_code = visualization_code[:-3]
            visualization_code = visualization_code.strip()
            
            return {
                'success': True,
                'visualization_code': visualization_code,
                'model_used': 'gpt-4o-code-interpreter',
                'tokens_used': None
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization code: {e}")
            return {
                'success': False,
                'error': str(e),
                'visualization_code': None
            }
    
    def _create_sql_generation_prompt(self, user_query: str, database_schemas: Dict[str, Any], 
                                    normalized_query: str = None) -> str:
        """Create prompt for SQL generation"""
        prompt = f"""
        Generate a PostgreSQL SQL query based on the following information:
        
        User Query: {user_query}
        """
        
        if normalized_query and normalized_query != user_query:
            prompt += f"\nNormalized Query: {normalized_query}"
        
        prompt += f"\n\nDatabase Schema:\n"
        for table_name, schema in database_schemas.items():
            prompt += f"\nTable: {table_name}\n"
            for column in schema.get('columns', []):
                prompt += f"  - {column['name']}: {column['type']}"
                if not column['nullable']:
                    prompt += " (NOT NULL)"
                
                # Add sample values if available
                if 'sample_values' in column and column['sample_values']:
                    sample_str = ", ".join(column['sample_values'])
                    prompt += f" (examples: {sample_str})"
                prompt += "\n"
        
        prompt += f"""
        
        Instructions:
        1. Generate only the SQL query, no explanations
        2. Use proper PostgreSQL syntax
        3. Include appropriate WHERE clauses based on the user query
        4. Use table aliases for readability
        5. Limit results to reasonable amounts (e.g., LIMIT 100)
        6. Use the sample values (examples) to understand what data looks like in each column
        7. When filtering by specific values, use the actual sample values as reference
        
        SQL Query:"""
        
        return prompt
    
    def _create_response_generation_prompt(self, user_query: str, sql_query: str, 
                                        query_results: Any, database_schemas: Dict[str, Any]) -> str:
        """Create prompt for natural language response generation"""
        # Convert query results to string representation
        if hasattr(query_results, 'shape'):
            results_summary = f"DataFrame with {query_results.shape[0]} rows and {query_results.shape[1]} columns"
            if query_results.shape[0] > 0:
                results_summary += f"\nFirst few rows:\n{query_results.head().to_string()}"
        else:
            results_summary = str(query_results)
        
        # prompt = f"""
        # Explain the following query results in natural language:
        #
        # Original Query: {user_query}
        # SQL Query: {sql_query}
        # Results: {results_summary}
        #
        # Instructions:
        # 1. Explain what the query was asking for
        # 2. Summarize the key findings from the results
        # 3. Provide insights about the data
        # 4. Use clear, non-technical language
        # 5. Keep the response concise but informative
        # 6. Provide the response in the same language as user
        #
        # Response:"""

        prompt = f"""Explain these query results in natural language:
        User Question: {user_query}
        Results: {results_summary}
        
        Instructions:
        - Show the actual data and numbers
        - Keep response concise (2-3 sentences)
        - Use the same language as the user
        - Focus on key findings
        
        Response:"""
        
        return prompt
    
    def _generate_fallback_response(self, user_query: str, query_results: Any) -> Dict[str, Any]:
        """Generate a simple fallback response for large datasets without LLM"""
        try:
            if hasattr(query_results, 'shape'):
                rows, cols = query_results.shape
                response = f"Query returned {rows} rows with {cols} columns. "
                
                # Add basic statistics if possible
                if rows > 0:
                    numeric_cols = query_results.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        response += f"Data includes numeric columns: {', '.join(numeric_cols[:3])}. "
                    
                    # Show first few values if not too many
                    if rows <= 10:
                        response += f"Results: {query_results.to_dict('records')[:3]}"
                    else:
                        response += "Results are available in the data table below."
            else:
                response = f"Query executed successfully. Results: {str(query_results)[:200]}..."
            
            return {
                'success': True,
                'natural_response': response,
                'model_used': 'fallback',
                'tokens_used': 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'natural_response': f"Query executed successfully. {len(query_results) if hasattr(query_results, '__len__') else 'Some'} results returned."
            }
    
    def _generate_code_interpreter_chart(self, query: str, df) -> str:
        """Generate chart code using OpenAI Code Interpreter approach"""
        try:
            # Prepare data summary for the Code Interpreter
            data_summary = self._prepare_data_summary(df)
            
            # Create a comprehensive prompt for Code Interpreter
            prompt = f"""
            You are a data visualization expert. Generate Python code to create a chart based on the user's query and data.

            USER QUERY: "{query}"

            DATA SUMMARY:
            {data_summary}

            REQUIREMENTS:
            1. Create a Python function called `create_chart(data)` that takes the data as parameter. 
            2. Use plotly.graph_objects (go) for visualization
            3. Make sure all required packages are imported
            4. Handle edge cases (empty data, insufficient columns)
            5. Create the most appropriate chart type based on the query and data
            6. Make the chart title and labels relevant to the user's query
            8. Return a valid Plotly figure object
            9. Label and text in the chart to same language as user query if user query  is other than English
            10. IMPORTANT: Use ALL the data in the dataset, not just a sample - show complete results
            11. If there are many data points, consider using appropriate chart types that can handle large datasets
            12. Also add the code to call that function with the data provided in data summary.

            CHART TYPE GUIDELINES:
            - For comparisons between categories: use bar charts
            - For trends over time: use line charts  
            - For distributions/percentages: use pie charts
            - For correlations between two numeric values: use scatter plots
            - For rankings: use horizontal bar charts
            - For geographic data: use map visualizations
            - For multi-dimensional data: use heatmaps or treemaps

            Generate ONLY the Python code, no explanations or markdown formatting.
            """
            # 7. Use proper error handling with try-catch blocks and import Exception
            # Use GPT-4o for better code generation (Code Interpreter approach)
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better code generation
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Generate Python code using plotly for creating charts. You have access to code execution capabilities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent code generation
                max_tokens=2000
            )

            chart_code = response.choices[0].message.content.strip()

            # Clean up the response
            if "```python" in chart_code:
                chart_code = chart_code.split("```python")[1].split("```")[0].strip()
            elif "```" in chart_code:
                chart_code = chart_code.split("```")[1].strip()

            # Ensure required imports are present
            if "import plotly.graph_objects as go" not in chart_code:
                chart_code = "import plotly.graph_objects as go\n" + chart_code
            if "import pandas as pd" not in chart_code:
                chart_code = "import pandas as pd\n" + chart_code

            logger.info(f"Code Interpreter generated chart code: {chart_code[:200]}...")
            return chart_code

        except Exception as e:
            logger.error(f"Error generating chart code with Code Interpreter: {e}")
            return self._get_fallback_chart_code()
    
    def _prepare_data_summary(self, df) -> str:
        """Prepare a comprehensive data summary for Code Interpreter"""
        try:
            summary = f"""
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Columns: {list(df.columns)}
            - Data types: {dict(df.dtypes)}
            """
            
            # Add numeric columns info
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                summary += f"\n- Numeric columns: {numeric_cols}"
                for col in numeric_cols:  # Show stats for ALL numeric columns
                    summary += f"\n  - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
            
            # Add categorical columns info
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                summary += f"\n- Categorical columns: {categorical_cols}"
                for col in categorical_cols:  # Show unique values for ALL categorical columns
                    unique_vals = df[col].unique()[:10]  # First 10 unique values
                    summary += f"\n  - {col}: {len(df[col].unique())} unique values, sample: {list(unique_vals)}"
            
            # Add sample data - show more rows for better chart generation
            if not df.empty:
                # Show up to 10 rows or all rows if less than 10
                sample_rows = min(10, len(df))
                summary += f"\n- Sample data (first {sample_rows} rows):\n{df.head(sample_rows).to_string()}"
                
                # If there are more rows, mention the total
                if len(df) > 10:
                    summary += f"\n- Total rows in dataset: {len(df)}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error preparing data summary: {e}")
            return f"Data shape: {df.shape}, Columns: {list(df.columns)}"
    
    def _get_fallback_chart_code(self) -> str:
        """Fallback chart code when Code Interpreter fails"""
        return """
def create_chart(data):
    import plotly.graph_objects as go
    import pandas as pd
    
    try:
        # Basic error handling for empty data
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available for visualization", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create a simple bar chart as fallback
        fig = go.Figure()
        fig.add_annotation(text="Chart generation failed - using fallback", 
                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", 
                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
"""
    
    # def _create_visualization_prompt(self, user_query: str, query_results: Any,
    #                                database_schemas: Dict[str, Any]) -> str:
    #     """Create prompt for visualization code generation"""
    #     # Convert query results to string representation
    #     if hasattr(query_results, 'shape'):
    #         results_summary = f"DataFrame with {query_results.shape[0]} rows and {query_results.shape[1]} columns"
    #         if query_results.shape[0] > 0:
    #             results_summary += f"\nColumns: {list(query_results.columns)}"
    #             results_summary += f"\nData types: {dict(query_results.dtypes)}"
    #     else:
    #         results_summary = str(query_results)
    #
    #     prompt = f"""
    #     Generate Python code to create a visualization for the following data:
    #
    #     User Query: {user_query}
    #     Data: {results_summary}
    #
    #     Instructions:
    #     1. Use plotly for the visualization
    #     2. Choose the most appropriate chart type (bar, line, pie, scatter, etc.)
    #     3. Include proper titles and labels
    #     4. Handle the data appropriately (aggregate if needed)
    #     5. Make the visualization interactive and informative
    #     6. Include error handling for edge cases
    #     7. Use bright, visible colors and a light theme
    #     8. Set template='plotly_white' for better visibility
    #     9. Use color_discrete_sequence or color_continuous_scale for better colors
    #     10. Ensure the figure variable is named 'fig'
    #
    #     Python Code:"""
    #
    #     return prompt


    def _format_schema_for_llm(self, database_schema: Dict[str, Any]) -> str:
        """Format database schema for LLM consumption"""
        schema_text = ""
        for table_name, table_info in database_schema.items():
            schema_text += f"\nTable: {table_name}\n"
            schema_text += "Columns:\n"
            for column in table_info.get('columns', []):
                nullable = "NULL" if column.get('nullable') else "NOT NULL"
                schema_text += f"  - {column['name']}: {column['type']} ({nullable}) (Sample Values: {column['sample_values']})\n"

            if table_info.get('primary_key'):
                schema_text += f"Primary Key: {', '.join(table_info['primary_key'])}\n"

            if table_info.get('foreign_keys'):
                schema_text += "Foreign Keys:\n"
                for fk in table_info['foreign_keys']:
                    schema_text += f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"

            schema_text += "\n"

        return schema_text

    def _format_values_for_llm(self, relevant_values: List[Dict[str, Any]]) -> str:
        """Format relevant categorical values for LLM consumption"""
        if not relevant_values:
            return ""
        
        values_text = "Keyword Replacements and Database Values:\n"
        for value_info in relevant_values:
            if value_info.get('type') == 'keyword_replacement_with_options':
                # New format: keyword replacement with multiple options
                original = value_info['original_keyword']
                values_text += f"- Keyword '{original}' has multiple similar matches:\n"
                
                for i, match in enumerate(value_info.get('similar_matches', []), 1):
                    keyword = match['keyword']
                    confidence = match['confidence']
                    database_values = match.get('database_values', [])
                    
                    values_text += f"  {i}. '{keyword}' (confidence: {confidence:.3f})"
                    if database_values:
                        values_text += f" - Database values: {', '.join(database_values)}"
                    values_text += "\n"
                
                values_text += f"  → Best match used: '{value_info.get('best_match', 'N/A')}'\n"
                values_text += f"  → Choose the most appropriate match based on context\n"
                
            elif value_info.get('type') == 'keyword_replacement':
                # Legacy format: single keyword replacement
                values_text += f"- Keyword '{value_info['original_keyword']}' was replaced with '{value_info['replaced_with']}' (confidence: {value_info['confidence']:.3f})\n"
                if value_info.get('database_values'):
                    values_text += f"  Available database values: {', '.join(value_info['database_values'])}\n"
            else:
                # Fallback for old format or other types
                table_name = value_info.get('table_name', 'Unknown')
                column_name = value_info.get('column_name', 'Unknown')
                value = value_info.get('value', 'Unknown')
                values_text += f"- Table: {table_name}, Column: {column_name}, Value: {value}\n"
        
        return values_text

    def validate_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Basic validation of SQL query syntax"""
        try:
            # Basic checks for common SQL issues
            sql_lower = sql_query.lower()

            # Check for basic SQL structure
            if not any(keyword in sql_lower for keyword in ['select', 'from']):
                return False

            # Check for balanced parentheses
            if sql_lower.count('(') != sql_lower.count(')'):
                return False

            # Check for proper table references
            if 'from' in sql_lower and not any(
                    table in sql_lower for table in ['private_sector_contributor_distribution_by_legal_entity',
                                                     'private_sector_contributor_distribution_by_economic_activity',
                                                     'private_sector_contributor_distribution_by_occupation_group']):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating SQL query: {e}")
            return False

        # """Validate SQL query using LLM"""
        # try:
        #     prompt = f"""
        #     Validate the following PostgreSQL SQL query for syntax and best practices:
        #
        #     {sql_query}
        #
        #     Instructions:
        #     1. Check for syntax errors
        #     2. Identify potential issues
        #     3. Suggest improvements
        #     4. Return only 'VALID' or 'INVALID' followed by explanation
        #
        #     Validation:"""
        #
        #     response = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=[
        #             {"role": "system", "content": "You are a SQL expert. Validate SQL queries and provide feedback."},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=0.1,
        #         max_tokens=200
        #     )
        #
        #     validation_result = response.choices[0].message.content.strip()
        #
        #     return {
        #         'success': True,
        #         'validation_result': validation_result,
        #         'is_valid': 'VALID' in validation_result.upper(),
        #         'model_used': self.model
        #     }
        #
        # except Exception as e:
        #     logger.error(f"Error validating SQL query: {e}")
        #     return {
        #         'success': False,
        #         'error': str(e),
        #         'validation_result': None
        #     }


    def _analyze_data_structure(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data structure to understand characteristics"""
        try:
            if data.empty:
                return {}

            insights = {}
            columns = list(data.iloc[0].keys())
            insights['total_columns'] = len(columns)
            insights['total_rows'] = len(data)

            # Analyze column types
            numeric_columns = []
            categorical_columns = []
            sequential_columns = []

            for col in columns:
                sample_values = [row for row in data[col].dropna().head(10).tolist()]
                if not sample_values:
                    continue

                # Check if numeric
                try:
                    float(sample_values[0])
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    categorical_columns.append(col)

                # Check if sequential (dates, numbers in order)
                if len(sample_values) > 1:
                    try:
                        # Try to detect if values are sequential
                        if all(isinstance(v, (int, float)) for v in sample_values):
                            sorted_values = sorted(sample_values)
                            if sorted_values == sample_values or sorted_values == sample_values[::-1]:
                                sequential_columns.append(col)
                    except:
                        pass

            insights['numeric_columns'] = numeric_columns
            insights['categorical_columns'] = categorical_columns
            insights['sequential_columns'] = sequential_columns
            insights['has_numeric_values'] = len(numeric_columns) > 0
            insights['has_categorical_values'] = len(categorical_columns) > 0
            insights['has_sequential_values'] = len(sequential_columns) > 0
            insights['has_two_numeric_columns'] = len(numeric_columns) >= 2

            return insights

        except Exception as e:
            logger.error(f"Error analyzing data structure: {e}")
            return {}

    def _generate_fallback_chart_code(self, columns: List[str], data_insights: Dict[str, Any], query: str) -> str:
        """Generate fallback chart code when LLM fails"""
        try:
            # Simple fallback to bar chart
            return f'''
        import plotly.graph_objects as go
        import pandas as pd

        def create_chart(data):
            try:
                if not data or len(data) == 0:
                    fig = go.Figure()
                    fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                    return fig

                df = pd.DataFrame(data)
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    fig = go.Figure(data=[go.Bar(x=df[x_col], y=df[y_col])])
                    fig.update_layout(
                        title=f"{{y_col}} by {{x_col}}",
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        template="plotly_white",
                        title_x=0.5
                    )
                else:
                    fig = go.Figure()
                    fig.add_annotation(text="Insufficient data for chart", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
            except Exception as e:
                fig = go.Figure()
                fig.add_annotation(text=f"Error: {{str(e)}}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
        '''
        except Exception as e:
            logger.error(f"Error generating fallback chart code: {e}")
            return None