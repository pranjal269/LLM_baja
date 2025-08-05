from typing import List, Dict, Any, Optional
from app.services.embeddings import EmbeddingService
from app.services.query_parser import QueryParser
from app.models import SearchResult, DocumentChunk, EntityExtraction
import logging

logger = logging.getLogger(__name__)

class SemanticSearch:
    """Handles semantic search operations"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.query_parser = QueryParser()
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform semantic search with entity extraction"""
        try:
            # Extract entities from query
            entities = self.query_parser.extract_entities(query)
            
            # Generate multiple search queries
            search_queries = self.query_parser.generate_search_queries(query, entities)
            
            # Perform searches for each query
            all_results = []
            seen_chunk_ids = set()
            
            for search_query in search_queries[:3]:  # Limit to top 3 queries
                results = self.embedding_service.search_similar_chunks(
                    query=search_query,
                    top_k=top_k,
                    filters=filters
                )
                
                for result in results:
                    chunk_id = result["chunk_id"]
                    if chunk_id not in seen_chunk_ids:
                        search_result = self._create_search_result(result)
                        if search_result:
                            all_results.append(search_result)
                            seen_chunk_ids.add(chunk_id)
            
            # Sort by similarity score and return top results
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _create_search_result(self, pinecone_result: Dict[str, Any]) -> Optional[SearchResult]:
        """Convert Pinecone result to SearchResult object"""
        try:
            metadata = pinecone_result["metadata"]
            
            chunk = DocumentChunk(
                chunk_id=pinecone_result["chunk_id"],
                document_name=metadata.get("document_name", ""),
                chunk_text=pinecone_result["chunk_text"],
                chunk_index=metadata.get("chunk_index", 0),
                page_number=metadata.get("page_number"),
                metadata=metadata
            )
            
            return SearchResult(
                chunk=chunk,
                similarity_score=pinecone_result["similarity_score"]
            )
            
        except Exception as e:
            logger.error(f"Error creating search result: {str(e)}")
            return None
    
    def search_with_reranking(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search with additional reranking based on entities"""
        try:
            # Get initial results
            results = self.search(query, top_k * 2, filters)  # Get more results for reranking
            
            # Extract entities for reranking
            entities = self.query_parser.extract_entities(query)
            
            # Rerank results based on entity relevance
            reranked_results = self._rerank_by_entities(results, entities)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search with reranking: {str(e)}")
            return []
    
    def _rerank_by_entities(self, results: List[SearchResult], entities: EntityExtraction) -> List[SearchResult]:
        """Rerank search results based on entity relevance"""
        for result in results:
            chunk_text_lower = result.chunk.chunk_text.lower()
            bonus_score = 0
            
            # Boost score for procedure mentions
            if entities.procedure and entities.procedure.lower() in chunk_text_lower:
                bonus_score += 0.1
            
            # Boost score for location mentions
            if entities.location and entities.location.lower() in chunk_text_lower:
                bonus_score += 0.05
            
            # Boost score for age-related terms
            if entities.age:
                age_terms = ["age", "years", "old", str(entities.age)]
                if any(term in chunk_text_lower for term in age_terms):
                    bonus_score += 0.05
            
            # Boost score for policy duration terms
            if entities.policy_duration:
                duration_terms = ["month", "year", "waiting", "period"]
                if any(term in chunk_text_lower for term in duration_terms):
                    bonus_score += 0.05
            
            # Apply bonus (but cap the total score at 1.0)
            result.similarity_score = min(1.0, result.similarity_score + bonus_score)
        
        # Sort by updated scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    def search_by_document(self, query: str, document_name: str, top_k: int = 10) -> List[SearchResult]:
        """Search within a specific document"""
        filters = {"document_name": document_name}
        return self.search(query, top_k, filters)
    
    def get_context_chunks(self, chunk_id: str, context_size: int = 2) -> List[DocumentChunk]:
        """Get surrounding chunks for better context"""
        try:
            # Extract document name and chunk index from chunk_id
            # Assuming chunk_id format: "document_name_index_uuid"
            parts = chunk_id.split('_')
            if len(parts) < 3:
                return []
            
            document_name = '_'.join(parts[:-2])
            chunk_index = int(parts[-2])
            
            # Search for surrounding chunks
            surrounding_chunks = []
            for i in range(max(0, chunk_index - context_size), chunk_index + context_size + 1):
                # This is a simplified approach - in a real implementation,
                # you might want to store chunk relationships more explicitly
                filters = {
                    "document_name": document_name,
                    "chunk_index": i
                }
                
                results = self.embedding_service.search_similar_chunks(
                    query="",  # Empty query to get any chunk matching filters
                    top_k=1,
                    filters=filters
                )
                
                if results:
                    result = results[0]
                    chunk = DocumentChunk(
                        chunk_id=result["chunk_id"],
                        document_name=result["metadata"]["document_name"],
                        chunk_text=result["chunk_text"],
                        chunk_index=result["metadata"]["chunk_index"],
                        page_number=result["metadata"].get("page_number"),
                        metadata=result["metadata"]
                    )
                    surrounding_chunks.append(chunk)
            
            # Sort by chunk index
            surrounding_chunks.sort(key=lambda x: x.chunk_index)
            return surrounding_chunks
            
        except Exception as e:
            logger.error(f"Error getting context chunks: {str(e)}")
            return []