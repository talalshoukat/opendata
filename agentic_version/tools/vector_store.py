import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """Simplified vector store for keyword normalization and similarity search using scikit-learn"""
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.vectors = None
        self.keywords = []
        self.keyword_metadata = []
        self.is_fitted = False
        
        # Load existing store if available
        self.load_store()
    
    def add_categorical_values(self, table_name: str, categorical_values: Dict[str, List[str]]):
        """Add categorical values from a table to the vector store"""
        try:
            for column_name, values in categorical_values.items():
                for value in values:
                    if value and str(value).strip():
                        # Create metadata for this keyword
                        metadata = {
                            'table_name': table_name,
                            'column_name': column_name,
                            'original_value': str(value),
                            'type': 'categorical_value'
                        }
                        
                        # Add to keywords list
                        self.keywords.append(str(value))
                        self.keyword_metadata.append(metadata)
            
            logger.info(f"Added {len(categorical_values)} categorical columns from {table_name}")
        except Exception as e:
            logger.error(f"Error adding categorical values from {table_name}: {e}")
    
    def build_index(self):
        """Build the vector index from keywords using scikit-learn"""
        try:
            if not self.keywords:
                logger.warning("No keywords to build index from")
                return
            
            # Create TF-IDF vectors
            self.vectors = self.vectorizer.fit_transform(self.keywords)
            self.is_fitted = True
            logger.info(f"Built vector index with {len(self.keywords)} keywords")
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            raise
    
    def search_similar_keywords(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar keywords using vector similarity"""
        try:
            if not self.is_fitted or self.vectors is None:
                logger.warning("Index not built, building now...")
                self.build_index()
            
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= threshold and idx < len(self.keywords):
                    result = {
                        'keyword': self.keywords[idx],
                        'score': float(score),
                        'metadata': self.keyword_metadata[idx] if idx < len(self.keyword_metadata) else {}
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar keywords: {e}")
            return []
    
    def normalize_query_keywords(self, query: str) -> Dict[str, Any]:
        """Normalize keywords in a query using vector similarity"""
        try:
            # Extract potential keywords from the query
            words = query.lower().split()
            potential_keywords = [word for word in words if len(word) > 2]
            
            normalized_keywords = []
            replacements_made = []
            
            for keyword in potential_keywords:
                # Search for similar keywords
                similar = self.search_similar_keywords(keyword, top_k=3, threshold=0.6)
                
                if similar:
                    best_match = similar[0]
                    if best_match['score'] >= Config.SIMILARITY_THRESHOLD:
                        replacement = {
                            'original_keyword': keyword,
                            'replaced_with': best_match['keyword'],
                            'confidence': best_match['score'],
                            'metadata': best_match['metadata']
                        }
                        replacements_made.append(replacement)
                        normalized_keywords.append(best_match['keyword'])
                    else:
                        normalized_keywords.append(keyword)
                else:
                    normalized_keywords.append(keyword)
            
            # Create normalized query
            normalized_query = query
            for replacement in replacements_made:
                normalized_query = normalized_query.replace(
                    replacement['original_keyword'], 
                    replacement['replaced_with']
                )
            
            return {
                'original_query': query,
                'normalized_query': normalized_query,
                'replacements': replacements_made,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error normalizing query keywords: {e}")
            return {
                'original_query': query,
                'normalized_query': query,
                'replacements': [],
                'success': False,
                'error': str(e)
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_keywords': len(self.keywords),
            'is_fitted': self.is_fitted,
            'index_size': self.vectors.shape[0] if self.vectors is not None else 0,
            'tables_processed': len(set(meta['table_name'] for meta in self.keyword_metadata))
        }
    
    def save_store(self, path: Optional[str] = None):
        """Save the vector store to disk"""
        try:
            if path is None:
                path = Config.VECTOR_STORE_PATH
            
            os.makedirs(path, exist_ok=True)
            
            # Save keywords and metadata
            with open(os.path.join(path, 'keywords.pkl'), 'wb') as f:
                pickle.dump(self.keywords, f)
            
            with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(self.keyword_metadata, f)
            
            # Save vectorizer
            with open(os.path.join(path, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save vectors
            if self.vectors is not None:
                with open(os.path.join(path, 'vectors.pkl'), 'wb') as f:
                    pickle.dump(self.vectors, f)
            
            logger.info(f"Vector store saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_store(self, path: Optional[str] = None):
        """Load the vector store from disk"""
        try:
            if path is None:
                path = Config.VECTOR_STORE_PATH
            
            if not os.path.exists(path):
                logger.info("Vector store path does not exist, starting fresh")
                return
            
            # Load keywords and metadata
            keywords_path = os.path.join(path, 'keywords.pkl')
            metadata_path = os.path.join(path, 'metadata.pkl')
            vectorizer_path = os.path.join(path, 'vectorizer.pkl')
            vectors_path = os.path.join(path, 'vectors.pkl')
            
            if all(os.path.exists(p) for p in [keywords_path, metadata_path, vectorizer_path, vectors_path]):
                with open(keywords_path, 'rb') as f:
                    self.keywords = pickle.load(f)
                
                with open(metadata_path, 'rb') as f:
                    self.keyword_metadata = pickle.load(f)
                
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(vectors_path, 'rb') as f:
                    self.vectors = pickle.load(f)
                
                self.is_fitted = True
                
                logger.info(f"Loaded vector store with {len(self.keywords)} keywords")
            else:
                logger.info("Some vector store files missing, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Continue with fresh store
            pass
    
    def reset_store(self):
        """Reset the vector store completely"""
        self.keywords = []
        self.keyword_metadata = []
        self.vectors = None
        self.is_fitted = False
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        logger.info("Vector store reset completed")
