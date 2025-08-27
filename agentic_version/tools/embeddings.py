import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from config.config import Config
from tools.embedding_cache import EmbeddingCache

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
    """OpenAI embedding provider using OpenAI's embedding API with caching support"""
    
    def __init__(self, model: str = None, cache_enabled: bool = None):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model or Config.OPENAI_EMBEDDING_MODEL
        self.is_fitted = False
        self.embedding_dimension = None
        self.cache = EmbeddingCache(enabled=cache_enabled)
    
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
        """Get embeddings from OpenAI API with caching support"""
        try:
            # Handle empty texts
            if not texts:
                return []
            
            # Filter out empty strings
            non_empty_texts = [text for text in texts if text and text.strip()]
            if not non_empty_texts:
                return []
            
            # Check cache first
            cached_embeddings = {}
            texts_to_fetch = []
            
            for text in non_empty_texts:
                cached_embedding = self.cache.get_embedding(text, self.model)
                if cached_embedding is not None:
                    cached_embeddings[text] = cached_embedding
                    logger.debug(f"Using cached embedding for: {text[:50]}...")
                else:
                    texts_to_fetch.append(text)
            
            # Fetch missing embeddings from API
            api_embeddings = []
            if texts_to_fetch:
                logger.info(f"Fetching {len(texts_to_fetch)} embeddings from OpenAI API")
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_fetch
                )
                
                api_embeddings = [data.embedding for data in response.data]
                
                # Store new embeddings in cache
                for text, embedding in zip(texts_to_fetch, api_embeddings):
                    self.cache.store_embedding(text, self.model, embedding)
                    logger.debug(f"Stored new embedding in cache for: {text[:50]}...")
            
            # Combine cached and new embeddings in original order
            final_embeddings = []
            api_index = 0
            
            for text in non_empty_texts:
                if text in cached_embeddings:
                    final_embeddings.append(cached_embeddings[text])
                else:
                    final_embeddings.append(api_embeddings[api_index])
                    api_index += 1
            
            return final_embeddings
            
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {e}")
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_cache_stats()
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear_cache()


def create_embedding_provider(embedding_type: str = None) -> EmbeddingProvider:
    """Factory function to create embedding provider based on configuration"""
    embedding_type = embedding_type or Config.EMBEDDING_TYPE
    
    if embedding_type.lower() == 'tfidf':
        return TfidfEmbeddingProvider()
    elif embedding_type.lower() == 'openai':
        return OpenAIEmbeddingProvider()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}. Supported types: 'tfidf', 'openai'")
