import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the agentic AI system"""
    
    # Database configuration
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'open_data'),
        'user': os.getenv('DB_USER', 'talal'),
        'password': os.getenv('DB_PASSWORD', 'my_password')
    }
    
    # OpenAI configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")
    
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    
    # Vector store configuration
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', './vector_store')
    
    # Embedding configuration - Using OpenAI by default for better keyword matching
    EMBEDDING_TYPE = os.getenv('EMBEDDING_TYPE', 'openai')  # 'tfidf', 'openai', 'sentence_transformers'
    OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')  # Latest model for better accuracy
    
    # Alternative embedding models
    SENTENCE_TRANSFORMERS_MODEL = os.getenv('SENTENCE_TRANSFORMERS_MODEL', 'all-MiniLM-L6-v2')
    
    # Table names (from the original system)
    TABLES = [
        'private_sector_contributor_distribution_by_legal_entity',
        'private_sector_contributor_distribution_by_economic_activity',
        'private_sector_contributor_distribution_by_occupation_group',
        'annuity_benefit',
        'establishments_by_region',
        'contributors_by_nationality',
        'total_beneficiaries'
    ]
    
    # Agent configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    MAX_TOOL_CALLS = int(os.getenv('MAX_TOOL_CALLS', '10'))
    
    # LLM configuration
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1000'))
    
    # Vector search configuration
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    MAX_CANDIDATES = int(os.getenv('MAX_CANDIDATES', '10'))
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get SQLAlchemy database URL"""
        return f"postgresql://{cls.DB_CONFIG['user']}:{cls.DB_CONFIG['password']}@{cls.DB_CONFIG['host']}:{cls.DB_CONFIG['port']}/{cls.DB_CONFIG['database']}"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        required_vars = ['OPENAI_API_KEY']
        for var in required_vars:
            if not getattr(cls, var, None):
                raise ValueError(f"Required configuration variable {var} is not set")
        return True
