import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from config.config import Config

# Optional imports for additional embedding providers
# try:
#     from sentence_transformers import SentenceTransformer
#     SENTENCE_TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     SENTENCE_TRANSFORMERS_AVAILABLE = False
#     logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Fit the embedding model on the given texts"""
        pass
    
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to embeddings"""
        pass
    
    @abstractmethod
    def get_similarity(self, query_embedding: np.ndarray, target_embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity between query and target embeddings"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider"""
        pass


class TfidfEmbeddingProvider(EmbeddingProvider):
    """TF-IDF based embedding provider using scikit-learn"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF vectorizer on the given texts"""
        try:
            self.vectorizer.fit(texts)
            self.is_fitted = True
            logger.info(f"TF-IDF vectorizer fitted on {len(texts)} texts")
        except Exception as e:
            logger.error(f"Error fitting TF-IDF vectorizer: {e}")
            raise
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors"""
        try:
            if not self.is_fitted:
                raise ValueError("TF-IDF vectorizer must be fitted before transform")
            
            vectors = self.vectorizer.transform(texts)
            return vectors.toarray()  # Convert sparse matrix to dense array
        except Exception as e:
            logger.error(f"Error transforming texts with TF-IDF: {e}")
            raise
    
    def get_similarity(self, query_embedding: np.ndarray, target_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and target embeddings"""
        try:
            # Ensure query_embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding, target_embeddings).flatten()
            return similarities
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of TF-IDF vectors"""
        if self.is_fitted:
            return self.vectorizer.transform(['dummy']).shape[1]
        return 10000  # max_features default


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using OpenAI's embedding API"""
    
    def __init__(self, model: str = None):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model or Config.OPENAI_EMBEDDING_MODEL
        self.is_fitted = False
        self.embedding_dimension = None
        
        # Performance optimizations
        self.batch_size = 100  # Process embeddings in batches
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def fit(self, texts: List[str]) -> None:
        """Fit is not needed for OpenAI embeddings, but we'll validate the model"""
        try:
            # Test the model with a small sample to get embedding dimension
            if texts:
                test_text = texts[0] if len(texts) > 0 else "test"
                test_embedding = self._get_embeddings([test_text])
                self.embedding_dimension = len(test_embedding[0])
            else:
                # Use a default test to get dimension
                test_embedding = self._get_embeddings(["test"])
                self.embedding_dimension = len(test_embedding[0])
            
            self.is_fitted = True
            logger.info(f"OpenAI embedding provider initialized with model {self.model}, dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI embedding provider: {e}")
            raise
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with batch processing and retry logic"""
        try:
            # Handle empty texts
            if not texts:
                return []
            
            # Filter out empty strings
            non_empty_texts = [text for text in texts if text and text.strip()]
            if not non_empty_texts:
                return []
            
            # Fetch embeddings from API in batches
            logger.info(f"Fetching {len(non_empty_texts)} embeddings from OpenAI API using model: {self.model}")
            embeddings = self._fetch_embeddings_batch(non_empty_texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {e}")
            raise
    
    def _fetch_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings in batches with retry logic"""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._fetch_single_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _fetch_single_batch(self, texts: List[str]) -> List[List[float]]:
        """Fetch a single batch of embeddings with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                logger.debug(f"Successfully fetched {len(texts)} embeddings using {self.model}")
                return [data.embedding for data in response.data]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {self.retry_delay}s: {e}")
                    import time
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} attempts failed for batch: {e}")
                    raise
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to OpenAI embeddings"""
        try:
            embeddings = self._get_embeddings(texts)
            if not embeddings:
                # Return zero vector if no embeddings
                return np.zeros((len(texts), self.embedding_dimension or 1536))
            
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error transforming texts with OpenAI embeddings: {e}")
            raise
    
    def get_similarity(self, query_embedding: np.ndarray, target_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and target embeddings"""
        try:
            # Ensure query_embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding, target_embeddings).flatten()
            return similarities
        except Exception as e:
            logger.error(f"Error calculating OpenAI embedding similarity: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of OpenAI embeddings"""
        if self.embedding_dimension:
            return self.embedding_dimension
        
        # Default dimensions for common OpenAI models
        model_dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        
        return model_dimensions.get(self.model, 1536)
    



# class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
#     """Sentence Transformers embedding provider for local embeddings"""
#
#     def __init__(self, model_name: str = None):
#         if not SENTENCE_TRANSFORMERS_AVAILABLE:
#             raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
#
#         self.model_name = model_name or Config.SENTENCE_TRANSFORMERS_MODEL
#         self.model = None
#         self.is_fitted = False
#         self.embedding_dimension = None
#
#     def fit(self, texts: List[str]) -> None:
#         """Load the sentence transformer model"""
#         try:
#             self.model = SentenceTransformer(self.model_name)
#             self.embedding_dimension = self.model.get_sentence_embedding_dimension()
#             self.is_fitted = True
#             logger.info(f"Sentence Transformers model {self.model_name} loaded, dimension: {self.embedding_dimension}")
#         except Exception as e:
#             logger.error(f"Error loading Sentence Transformers model: {e}")
#             raise
#
#     def transform(self, texts: List[str]) -> np.ndarray:
#         """Transform texts to embeddings using Sentence Transformers"""
#         try:
#             if not self.is_fitted:
#                 self.fit(texts)
#
#             embeddings = self.model.encode(texts, convert_to_numpy=True)
#             return embeddings
#         except Exception as e:
#             logger.error(f"Error transforming texts with Sentence Transformers: {e}")
#             raise
#
#     def get_similarity(self, query_embedding: np.ndarray, target_embeddings: np.ndarray) -> np.ndarray:
#         """Calculate cosine similarity between query and target embeddings"""
#         try:
#             if query_embedding.ndim == 1:
#                 query_embedding = query_embedding.reshape(1, -1)
#
#             similarities = cosine_similarity(query_embedding, target_embeddings).flatten()
#             return similarities
#         except Exception as e:
#             logger.error(f"Error calculating Sentence Transformers similarity: {e}")
#             raise
#
#     def get_embedding_dimension(self) -> int:
#         """Get the dimension of Sentence Transformers embeddings"""
#         return self.embedding_dimension or 384  # Default for all-MiniLM-L6-v2



def create_embedding_provider(embedding_type: str = None) -> EmbeddingProvider:
    """Factory function to create embedding provider based on configuration"""
    embedding_type = embedding_type or Config.EMBEDDING_TYPE
    
    if embedding_type.lower() == 'tfidf':
        return TfidfEmbeddingProvider()
    elif embedding_type.lower() == 'openai':
        return OpenAIEmbeddingProvider()
    elif embedding_type.lower() == 'sentence_transformers':
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, falling back to OpenAI")
            return OpenAIEmbeddingProvider()
        return SentenceTransformersEmbeddingProvider()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. Supported types: 'tfidf', 'openai', 'sentence_transformers'")
