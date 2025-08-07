import pinecone
# import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any
from app.models import DocumentChunk
from app.config import settings
import logging
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Handles document embeddings and Pinecone operations"""
    
    def __init__(self):
        # Initialize Gemini for embeddings (fallback solution)
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Initialize Pinecone
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            if not settings.PINECONE_API_KEY or settings.PINECONE_API_KEY == "your-pinecone-api-key-here":
                logger.warning("⚠️  Pinecone API key not configured. Some features may not work.")
                self.index = None
                return
            
            pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT)
            
            existing_indexes = pinecone.list_indexes()
            if settings.PINECONE_INDEX_NAME not in existing_indexes:
                pinecone.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric='cosine'
                )
                logger.info(f"Created Pinecone index: {settings.PINECONE_INDEX_NAME}")
            
            self.index = pinecone.Index(settings.PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            logger.warning("⚠️  Pinecone initialization failed. Vector storage features disabled.")
            self.index = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using a simple hash-based approach"""
        try:
            # Simple hash-based embedding as fallback
            # This is a very basic approach - in production you'd want proper semantic embeddings
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hex to numbers and normalize to create a 384-dimensional vector
            embedding = []
            for i in range(0, len(text_hash), 2):
                val = int(text_hash[i:i+2], 16) / 255.0  # Normalize to 0-1
                embedding.append(val)
            
            # Pad or truncate to match EMBEDDING_DIMENSION (384)
            while len(embedding) < settings.EMBEDDING_DIMENSION:
                embedding.extend(embedding[:min(len(embedding), settings.EMBEDDING_DIMENSION - len(embedding))])
            embedding = embedding[:settings.EMBEDDING_DIMENSION]
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            return [self.generate_embedding(text) for text in texts]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks in Pinecone"""
        try:
            if not self.index:
                logger.warning("⚠️  Pinecone not available. Cannot store chunks.")
                return False
                
            # Prepare data for upsert
            vectors_to_upsert = []
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk.chunk_text for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                metadata = {
                    "document_name": chunk.document_name,
                    "chunk_text": chunk.chunk_text[:1000],  # Limit text size in metadata
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    **chunk.metadata
                }
                
                vectors_to_upsert.append({
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert in batches to avoid size limits
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Successfully stored {len(chunks)} chunks in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunks in Pinecone: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            if not self.index:
                logger.warning("⚠️  Pinecone not available. Cannot search chunks.")
                return []
                
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Prepare filter for Pinecone query
            pinecone_filter = {}
            if filters:
                for key, value in filters.items():
                    if key in ["document_name", "page_number"]:
                        pinecone_filter[key] = value
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    "chunk_id": match.id,
                    "similarity_score": float(match.score),
                    "metadata": match.metadata,
                    "chunk_text": match.metadata.get("chunk_text", "")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def delete_document(self, document_name: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            if not self.index:
                logger.warning("⚠️  Pinecone not available. Cannot delete document.")
                return False
                
            # Query all chunks for this document
            results = self.index.query(
                vector=[0] * settings.EMBEDDING_DIMENSION,  # Dummy vector
                top_k=10000,  # Large number to get all chunks
                include_metadata=True,
                filter={"document_name": document_name}
            )
            
            # Extract IDs to delete
            ids_to_delete = [match.id for match in results.matches]
            
            if ids_to_delete:
                # Delete in batches
                batch_size = 1000
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i + batch_size]
                    self.index.delete(ids=batch)
                
                logger.info(f"Deleted {len(ids_to_delete)} chunks for document: {document_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        try:
            if not self.index:
                logger.warning("⚠️  Pinecone not available. Cannot get stats.")
                return {
                    "total_vector_count": 0,
                    "dimension": settings.EMBEDDING_DIMENSION,
                    "index_fullness": 0.0,
                    "status": "not_configured"
                }
                
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "status": "connected"
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {
                "total_vector_count": 0,
                "dimension": settings.EMBEDDING_DIMENSION,
                "index_fullness": 0.0,
                "status": "error"
            }