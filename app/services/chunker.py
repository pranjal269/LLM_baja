import tiktoken
from typing import List, Dict, Any
from app.models import DocumentChunk
from app.config import settings
import uuid

class DocumentChunker:
    """Handles intelligent document chunking for better semantic search"""
    
    def __init__(self):
        # Using cl100k_base encoding (GPT-4 tokenizer)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences while preserving context"""
        import re
        # Split by sentence endings but keep the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, document_name: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Create overlapping chunks from document text"""
        if metadata is None:
            metadata = {}
            
        sentences = self.split_by_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(
                    current_chunk.strip(), 
                    document_name, 
                    chunk_index, 
                    metadata
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = self.count_tokens(current_chunk)
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(), 
                document_name, 
                chunk_index, 
                metadata
            ))
        
        return chunks
    
    def _create_chunk(self, text: str, document_name: str, index: int, metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a DocumentChunk object"""
        chunk_id = f"{document_name}_{index}_{str(uuid.uuid4())[:8]}"
        
        # Try to determine page number from metadata
        page_number = None
        if "page_texts" in metadata:
            # Find which page this chunk likely belongs to
            page_number = self._find_page_number(text, metadata["page_texts"])
        
        return DocumentChunk(
            chunk_id=chunk_id,
            document_name=document_name,
            chunk_text=text,
            chunk_index=index,
            page_number=page_number,
            metadata=metadata
        )
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for the next chunk"""
        words = text.split()
        if len(words) <= self.chunk_overlap:
            return text
        
        overlap_words = words[-self.chunk_overlap:]
        return " ".join(overlap_words)
    
    def _find_page_number(self, chunk_text: str, page_texts: Dict[int, str]) -> int:
        """Find the most likely page number for a chunk"""
        max_overlap = 0
        best_page = None
        
        chunk_words = set(chunk_text.lower().split())
        
        for page_num, page_text in page_texts.items():
            page_words = set(page_text.lower().split())
            overlap = len(chunk_words.intersection(page_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_page = page_num
        
        return best_page
    
    def create_semantic_chunks(self, text: str, document_name: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Create chunks based on semantic boundaries (paragraphs, sections)"""
        if metadata is None:
            metadata = {}
        
        # Split by double newlines (paragraph breaks) or section headers
        import re
        sections = re.split(r'\n\s*\n|\n(?=[A-Z][^a-z]*:)', text)
        sections = [s.strip() for s in sections if s.strip()]
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_tokens = self.count_tokens(section)
            
            if section_tokens <= self.chunk_size:
                # Section fits in one chunk
                chunks.append(self._create_chunk(section, document_name, chunk_index, metadata))
                chunk_index += 1
            else:
                # Section too large, split further
                sub_chunks = self.create_chunks(section, document_name, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
        
        return chunks