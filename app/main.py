from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import time
import logging
from datetime import datetime

from app.config import settings
from app.models import (
    QueryRequest, DecisionResponse, DocumentUploadRequest, 
    HealthCheckResponse, ErrorResponse, HackrxRequest, HackrxResponse
)
from app.services.document_loader import DocumentLoader
from app.services.chunker import DocumentChunker
from app.services.embeddings import EmbeddingService
from app.services.semantic_search import SemanticSearch
from app.services.query_parser import QueryParser
from app.services.answer_generator import AnswerGenerator
from app.services.document_downloader import DocumentDownloader
from app.services.question_answerer import QuestionAnswerer
from app.services.auth import get_current_user, create_demo_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Document Processing System",
    description="AI-powered document processing and query system for insurance, legal, and compliance documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_loader = DocumentLoader()
chunker = DocumentChunker()
embedding_service = EmbeddingService()
semantic_search = SemanticSearch()
query_parser = QueryParser()
answer_generator = AnswerGenerator()
document_downloader = DocumentDownloader()
question_answerer = QuestionAnswerer()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Document Processing System",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "hackrx_main": "/hackrx/run",
            "upload": "/upload-document", 
            "legacy_query": "/query",
            "health": "/health",
            "demo_token": "/demo-token"
        },
        "hackrx_format": {
            "input": {"documents": "URL", "questions": ["array of questions"]},
            "output": {"answers": ["array of answers"]}
        }
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Pinecone connection
        pinecone_stats = embedding_service.get_index_stats()
        pinecone_status = "healthy" if pinecone_stats else "unhealthy"
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            services={
                "pinecone": pinecone_status,
                "gemini": "healthy",  # Assume healthy if no errors
                "embedding": "healthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/demo-token")
async def get_demo_token():
    """Get a demo token for testing"""
    token = create_demo_token()
    return {
        "access_token": token,
        "token_type": "bearer",
        "message": "Use this token in the Authorization header: Bearer <token>"
    }

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process a document"""
    try:
        # Validate file type
        allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "message/rfc822"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
        
        # Validate file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Determine file type
        file_type_map = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "message/rfc822": "email"
        }
        file_type = file_type_map.get(file.content_type)
        
        # Use provided document name or file name
        doc_name = document_name or file.filename
        
        # Load and parse document
        text_content, metadata = document_loader.load_document(content, file_type)
        
        # Clean text
        cleaned_text = document_loader.preprocess_text(text_content)
        
        # Create chunks
        chunks = chunker.create_semantic_chunks(cleaned_text, doc_name, metadata)
        
        # Store in Pinecone
        success = embedding_service.store_chunks(chunks)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document")
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_name": doc_name,
            "chunks_created": len(chunks),
            "file_type": file_type,
            "file_size": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/hackrx/run", response_model=HackrxResponse)
async def hackrx_run(
    request: HackrxRequest,
    current_user: dict = Depends(get_current_user)
):
    """Main endpoint for HackRX - processes document URL and answers questions"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing HackRX request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Step 1: Download document from URL
        document_content, file_extension = document_downloader.download_document(request.documents)
        
        if not document_content:
            raise HTTPException(status_code=400, detail="Failed to download document from provided URL")
        
        # Step 2: Determine file type and process document
        file_type = document_downloader.get_file_type_from_extension(file_extension)
        logger.info(f"Processing document as {file_type}")
        
        # Step 3: Load and parse document
        text_content, metadata = document_loader.load_document(document_content, file_type)
        cleaned_text = document_loader.preprocess_text(text_content)
        
        # Step 4: Create chunks for better processing
        chunks = chunker.create_semantic_chunks(cleaned_text, "downloaded_document", metadata)
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Step 5: Try to use vector search if available, otherwise use full text
        if embedding_service.index:
            # Store chunks temporarily for searching
            temp_success = embedding_service.store_chunks(chunks)
            
            if temp_success:
                # Use semantic search approach
                all_search_results = []
                for question in request.questions:
                    search_results = semantic_search.search_with_reranking(
                        query=question,
                        top_k=5,
                        filters={"document_name": "downloaded_document"}
                    )
                    all_search_results.extend(search_results)
                
                # Remove duplicates and get top results
                unique_results = {}
                for result in all_search_results:
                    if result.chunk.chunk_id not in unique_results:
                        unique_results[result.chunk.chunk_id] = result
                
                top_results = list(unique_results.values())[:15]  # Top 15 unique chunks
                answers = question_answerer.answer_questions(request.questions, top_results)
                
                # Clean up temporary storage
                try:
                    embedding_service.delete_document("downloaded_document")
                except:
                    pass  # Ignore cleanup errors
            else:
                # Fallback to full text processing
                answers = question_answerer.answer_questions_with_text(request.questions, cleaned_text)
        else:
            # Use full text processing when vector search is not available
            logger.info("Using full text processing (vector search not available)")
            answers = question_answerer.answer_questions_with_text(request.questions, cleaned_text)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Processed {len(request.questions)} questions in {processing_time}ms")
        
        return HackrxResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in HackRX processing: {str(e)}")
        # Return fallback answers on error
        fallback_answers = [
            "Unable to process the question due to technical difficulties." 
            for _ in request.questions
        ]
        return HackrxResponse(answers=fallback_answers)

@app.post("/query", response_model=DecisionResponse)
async def process_query_legacy(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Legacy endpoint for processing single queries (backward compatibility)"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing legacy query: {request.query}")
        
        # Extract entities from query
        entities = query_parser.extract_entities(request.query)
        logger.info(f"Extracted entities: {entities}")
        
        # Perform semantic search
        search_results = semantic_search.search_with_reranking(
            query=request.query,
            top_k=10,
            filters=request.filters
        )
        logger.info(f"Found {len(search_results)} relevant chunks")
        
        # Generate decision
        decision_response = answer_generator.generate_decision(
            query=request.query,
            search_results=search_results,
            entities=entities
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        decision_response.processing_time_ms = processing_time
        
        logger.info(f"Generated decision: {decision_response.decision} (confidence: {decision_response.confidence_score})")
        
        return decision_response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/search")
async def search_documents(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Search documents without generating a decision"""
    try:
        search_results = semantic_search.search(
            query=request.query,
            top_k=10,
            filters=request.filters
        )
        
        return {
            "query": request.query,
            "results": [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "document_name": result.chunk.document_name,
                    "chunk_text": result.chunk.chunk_text[:200] + "..." if len(result.chunk.chunk_text) > 200 else result.chunk.chunk_text,
                    "page_number": result.chunk.page_number,
                    "similarity_score": result.similarity_score
                }
                for result in search_results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.delete("/documents/{document_name}")
async def delete_document(
    document_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a document and all its chunks"""
    try:
        success = embedding_service.delete_document(document_name)
        
        if success:
            return {"message": f"Document '{document_name}' deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
            
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List all documents in the system"""
    try:
        stats = embedding_service.get_index_stats()
        return {
            "total_chunks": stats.get("total_vector_count", 0),
            "index_dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0)
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/explain")
async def explain_decision(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed explanation for a decision"""
    try:
        # Process the query to get decision
        entities = query_parser.extract_entities(request.query)
        search_results = semantic_search.search_with_reranking(
            query=request.query,
            top_k=10,
            filters=request.filters
        )
        decision_response = answer_generator.generate_decision(
            query=request.query,
            search_results=search_results,
            entities=entities
        )
        
        # Generate human-readable explanation
        explanation = answer_generator.generate_explanation(decision_response)
        
        return {
            "query": request.query,
            "decision": decision_response.decision,
            "explanation": explanation,
            "entities": entities,
            "supporting_chunks": [
                {
                    "document": result.chunk.document_name,
                    "text": result.chunk.chunk_text[:300] + "...",
                    "similarity": result.similarity_score,
                    "page": result.chunk.page_number
                }
                for result in search_results[:3]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error explaining decision: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error explaining decision: {str(e)}")

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return ErrorResponse(
        error="Internal Server Error",
        detail=str(exc),
        timestamp=datetime.now()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)