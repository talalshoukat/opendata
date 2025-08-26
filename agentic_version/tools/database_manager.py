import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.engine import URL
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for PostgreSQL operations"""
    
    def __init__(self):
        self.connection = None
        self.engine = None
        self.connect()
    
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            # Create SQLAlchemy engine
            url = URL.create(
                drivername="postgresql",
                username=Config.DB_CONFIG['user'],
                password=Config.DB_CONFIG['password'],
                host=Config.DB_CONFIG['host'],
                port=Config.DB_CONFIG['port'],
                database=Config.DB_CONFIG['database']
            )
            self.engine = create_engine(url)
            
            # Create psycopg2 connection for direct queries
            self.connection = psycopg2.connect(
                host=Config.DB_CONFIG['host'],
                port=Config.DB_CONFIG['port'],
                database=Config.DB_CONFIG['database'],
                user=Config.DB_CONFIG['user'],
                password=Config.DB_CONFIG['password']
            )
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a specific table"""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            
            schema_info = {
                'table_name': table_name,
                'columns': [],
                'primary_key': inspector.get_pk_constraint(table_name).get('constrained_columns', []),
                'foreign_keys': inspector.get_foreign_keys(table_name),
                'indexes': inspector.get_indexes(table_name)
            }
            
            for column in columns:
                schema_info['columns'].append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'default': column['default']
                })
            
            return schema_info
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return {}
    
    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schema information for all tables"""
        schemas = {}
        for table in Config.TABLES:
            schemas[table] = self.get_table_schema(table)
        return schemas
    
    def get_categorical_values(self, table_name: str, limit: int = 1000) -> Dict[str, List[str]]:
        """Extract categorical values from a table"""
        try:
            schema = self.get_table_schema(table_name)
            categorical_values = {}
            
            for column in schema['columns']:
                column_name = column['name']
                # Get unique values for each column (limited to avoid memory issues)
                query = f"""
                SELECT DISTINCT "{column_name}" 
                FROM "{table_name}" 
                WHERE "{column_name}" IS NOT NULL 
                LIMIT {limit}
                """
                
                df = pd.read_sql_query(query, self.engine)
                if not df.empty:
                    categorical_values[column_name] = df[column_name].astype(str).tolist()
            
            return categorical_values
        except Exception as e:
            logger.error(f"Error extracting categorical values from {table_name}: {e}")
            return {}
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame"""
        try:
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a table for context"""
        try:
            query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            logger.error(f"Error getting sample data from {table_name}: {e}")
            return pd.DataFrame()
    
    def validate_sql_query(self, query: str) -> Tuple[bool, str]:
        """Validate SQL query syntax without executing"""
        try:
            # Try to parse the query
            from sqlparse import parse
            parsed = parse(query)
            return True, "Query syntax is valid"
        except Exception as e:
            return False, f"Query syntax error: {str(e)}"
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table"""
        try:
            query = f'SELECT COUNT(*) FROM "{table_name}"'
            result = pd.read_sql_query(query, self.engine)
            return result.iloc[0, 0]
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            return 0
    
    def close(self):
        """Close database connections"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")
