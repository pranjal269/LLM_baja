from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentUploadRequest(BaseModel):
    document_type: str = Field(..., description="Type of document: pdf, docx, email")
    document_name: str = Field(..., description="Name of the document")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for search")

class HackrxRequest(BaseModel):
    documents: str = Field(..., description="URL to the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackrxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to each question")

class EntityExtraction(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    policy_type: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None

class ClauseReference(BaseModel):
    clause_id: str
    clause_text: str
    document_name: str
    page_number: Optional[int] = None
    confidence_score: float

class DecisionResponse(BaseModel):
    decision: str = Field(..., description="approved, rejected, or needs_review")
    amount: Optional[float] = Field(default=None, description="Payout amount if applicable")
    justification: str = Field(..., description="Explanation of the decision")
    referenced_clauses: List[ClauseReference] = Field(default=[], description="Clauses used for decision")
    extracted_entities: EntityExtraction
    confidence_score: float = Field(..., description="Overall confidence in the decision")
    processing_time_ms: int

class DocumentChunk(BaseModel):
    chunk_id: str
    document_name: str
    chunk_text: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = {}

class SearchResult(BaseModel):
    chunk: DocumentChunk
    similarity_score: float
    
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime