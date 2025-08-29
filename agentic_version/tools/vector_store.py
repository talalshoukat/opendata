import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import os
import re
from config.config import Config
from tools.embeddings import create_embedding_provider, EmbeddingProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """Simplified vector store for keyword normalization and similarity search using configurable embedding providers"""
    
    def __init__(self, embedding_type: str = None, openai_client=None):
        self.openai_client = openai_client
        # Use the configured embedding type if not specified
        if embedding_type is None:
            embedding_type = Config.EMBEDDING_TYPE
        self.embedding_provider = create_embedding_provider(embedding_type)
        self.vectors = None
        self.keywords = []
        self.keyword_metadata = []
        self.is_fitted = False
        
        logger.info(f"Initialized vector store with embedding type: {embedding_type}")
        logger.info(f"Embedding provider: {type(self.embedding_provider).__name__}")
        
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
        """Build the vector index from keywords using the configured embedding provider"""
        try:
            if not self.keywords:
                logger.warning("No keywords to build index from")
                return
            
            # Fit the embedding provider and create vectors
            self.embedding_provider.fit(self.keywords)
            self.vectors = self.embedding_provider.transform(self.keywords)
            self.is_fitted = True
            logger.info(f"Built vector index with {len(self.keywords)} keywords using {type(self.embedding_provider).__name__}")
            
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            raise
    
    def search_similar_keywords(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar keywords using vector similarity"""
        try:
            if not self.is_fitted or self.vectors is None:
                logger.warning("Index not built, building now...")
                self.build_index()
            
            # Vectorize the query using the embedding provider
            query_vector = self.embedding_provider.transform([query])
            
            # Calculate similarities using the embedding provider
            similarities = self.embedding_provider.get_similarity(query_vector, self.vectors)
            
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
                similar = self.search_similar_keywords(keyword, top_k=5, threshold=0.7)
                
                if similar:
                    # Filter matches above the similarity threshold
                    valid_matches = [match for match in similar if match['score'] >= Config.SIMILARITY_THRESHOLD]
                    
                    if valid_matches:
                        
                        # Create replacement with all valid matches for LLM to choose from
                        all_matches = []
                        for match in valid_matches:
                            # Get database values for each match (now cached)
                            database_values = self._get_database_values_for_keyword(match['keyword'])
                            
                            match_info = {
                                'keyword': match['keyword'],
                                'confidence': match['score'],
                                'metadata': match['metadata'],
                                'database_values': database_values
                            }
                            all_matches.append(match_info)
                        
                        replacement = {
                            'original_keyword': keyword,
                            'similar_matches': all_matches,
                            'best_match': valid_matches[0]['keyword'],  # Keep best match for backward compatibility
                            'confidence': valid_matches[0]['score'],
                            'type': 'keyword_replacement_with_options'
                        }
                        replacements_made.append(replacement)
                        
                        # Use the best match for the normalized query (LLM will see all options)
                        normalized_keywords.append(valid_matches[0]['keyword'])
                    else:
                        normalized_keywords.append(keyword)
                else:
                    normalized_keywords.append(keyword)
            
            # Create normalized query with better replacement logic
            normalized_query = query
            # for replacement in replacements_made:
            #     # Use word boundary replacement to avoid partial matches
            #     pattern = r'\b' + re.escape(replacement['original_keyword']) + r'\b'
            #     # Use best_match for the normalized query
            #     replacement_keyword = replacement.get('best_match', replacement.get('replaced_with', replacement['original_keyword']))
            #     normalized_query = re.sub(pattern, replacement_keyword, normalized_query, flags=re.IGNORECASE)
            
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
    
    def _get_database_values_for_keyword(self, keyword: str) -> List[str]:
        """Get all database values that match or are similar to the given keyword"""
        try:
            # Find all metadata entries that contain this keyword
            matching_values = []
            for i, meta in enumerate(self.keyword_metadata):
                if meta.get('original_value', '').lower() == keyword.lower():
                    matching_values.append(meta.get('original_value', ''))
                elif keyword.lower() in meta.get('original_value', '').lower():
                    matching_values.append(meta.get('original_value', ''))
            
            # Remove duplicates and return
            return list(set(matching_values))
            
        except Exception as e:
            logger.error(f"Error getting database values for keyword {keyword}: {e}")
            return []
    
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
            
            # Save embedding provider type and configuration (not the actual object to avoid pickle issues)
            embedding_info = {
                'type': type(self.embedding_provider).__name__,
                'model': getattr(self.embedding_provider, 'model', None),
                'is_fitted': getattr(self.embedding_provider, 'is_fitted', False),
                'embedding_dimension': getattr(self.embedding_provider, 'embedding_dimension', None)
            }
            with open(os.path.join(path, 'embedding_provider_info.pkl'), 'wb') as f:
                pickle.dump(embedding_info, f)
            
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
                logger.info(f"Vector store path {path} does not exist, starting fresh")
                return
            
            # Load keywords and metadata
            keywords_path = os.path.join(path, 'keywords.pkl')
            metadata_path = os.path.join(path, 'metadata.pkl')
            embedding_provider_info_path = os.path.join(path, 'embedding_provider_info.pkl')
            vectors_path = os.path.join(path, 'vectors.pkl')
            
            # Check for both old (vectorizer) and new (embedding_provider_info) formats
            vectorizer_path = os.path.join(path, 'vectorizer.pkl')
            old_embedding_provider_path = os.path.join(path, 'embedding_provider.pkl')
            has_old_format = os.path.exists(vectorizer_path)
            has_old_embedding_format = os.path.exists(old_embedding_provider_path)
            has_new_format = os.path.exists(embedding_provider_info_path)
            
            if all(os.path.exists(p) for p in [keywords_path, metadata_path, vectors_path]) and (has_old_format or has_old_embedding_format or has_new_format):
                with open(keywords_path, 'rb') as f:
                    self.keywords = pickle.load(f)
                
                with open(metadata_path, 'rb') as f:
                    self.keyword_metadata = pickle.load(f)
                
                # Load embedding provider based on available format
                if has_new_format:
                    # New format - recreate embedding provider from info
                    with open(embedding_provider_info_path, 'rb') as f:
                        embedding_info = pickle.load(f)
                    
                    # Recreate the embedding provider
                    if embedding_info['type'] == 'TfidfEmbeddingProvider':
                        from tools.embeddings import TfidfEmbeddingProvider
                        self.embedding_provider = TfidfEmbeddingProvider()
                        self.embedding_provider.is_fitted = embedding_info.get('is_fitted', False)
                    elif embedding_info['type'] == 'OpenAIEmbeddingProvider':
                        from tools.embeddings import OpenAIEmbeddingProvider
                        self.embedding_provider = OpenAIEmbeddingProvider(
                            model=embedding_info.get('model')
                        )
                        self.embedding_provider.is_fitted = embedding_info.get('is_fitted', False)
                        self.embedding_provider.embedding_dimension = embedding_info.get('embedding_dimension')
                    else:
                        # Fallback to default
                        self.embedding_provider = create_embedding_provider()
                        
                elif has_old_embedding_format:
                    # Try to load old embedding provider format (might fail due to pickle issues)
                    try:
                        with open(old_embedding_provider_path, 'rb') as f:
                            self.embedding_provider = pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load old embedding provider format: {e}, recreating...")
                        self.embedding_provider = create_embedding_provider()
                        
                elif has_old_format:
                    # Migrate from old format - load vectorizer and create TF-IDF provider
                    with open(vectorizer_path, 'rb') as f:
                        old_vectorizer = pickle.load(f)
                    # Create new TF-IDF provider and set the fitted vectorizer
                    from tools.embeddings import TfidfEmbeddingProvider
                    self.embedding_provider = TfidfEmbeddingProvider()
                    self.embedding_provider.vectorizer = old_vectorizer
                    self.embedding_provider.is_fitted = True
                
                with open(vectors_path, 'rb') as f:
                    self.vectors = pickle.load(f)
                
                self.is_fitted = True
                
                logger.info(f"Loaded vector store with {len(self.keywords)} keywords using {type(self.embedding_provider).__name__}")
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
        # Recreate the embedding provider with the configured type
        embedding_type = Config.EMBEDDING_TYPE
        self.embedding_provider = create_embedding_provider(embedding_type)
        logger.info(f"Vector store reset completed with embedding type: {embedding_type}")
