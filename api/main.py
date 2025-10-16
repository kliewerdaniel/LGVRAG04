"""
FastAPI application for the Graph + Vector RAG system.

This module provides REST API endpoints for querying the RAG system,
ingesting documents, and managing the knowledge base.
"""

import os
import logging
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_config
from db.ingest_data import DataIngester
from rag.retrieve import HybridRetriever
from rag.generate_answer import AnswerGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Graph + Vector RAG System API",
    description="A local RAG system combining graph-based knowledge representation with vector similarity search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system components
config = get_config()
ingester = DataIngester()
retriever = HybridRetriever()
answer_generator = AnswerGenerator()

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="The search query")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")
    include_entities: Optional[bool] = Field(True, description="Include entity information in response")

class QueryResponse(BaseModel):
    """Response model for queries."""
    query: str
    answer: str
    context_chunks: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    model_used: str
    generation_time: Optional[float]
    retrieval_stats: Dict[str, Any]

class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: Optional[str] = Field(None, description="Path to file to ingest")
    directory_path: Optional[str] = Field(None, description="Path to directory to ingest")

class IngestResponse(BaseModel):
    """Response model for ingestion."""
    success: bool
    stats: Dict[str, int]
    message: str

class SystemStats(BaseModel):
    """Response model for system statistics."""
    database_stats: Dict[str, int]
    embedding_dimension: int
    embedding_model: str
    llm_model: str
    api_status: str

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Graph + Vector RAG System API",
        "version": "1.0.0",
        "description": "A local RAG system combining graph-based knowledge representation with vector similarity search",
        "docs": "/docs"
    }

@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Query the RAG system and get an answer.

    This endpoint performs hybrid retrieval and generates an answer
    using the local LLM based on the retrieved context.
    """
    try:
        logger.info(f"Processing query: {request.query}")

        # Perform hybrid retrieval
        if request.include_entities:
            retrieval_results = retriever.retrieve_with_entities(request.query, request.top_k)
        else:
            chunks = retriever.retrieve(request.query, request.top_k)
            retrieval_results = {
                "query": request.query,
                "chunks": chunks,
                "entities": [],
                "relationships": []
            }

        # Generate answer using retrieved context
        answer = answer_generator.generate_answer_sync(
            query=request.query,
            context_chunks=retrieval_results["chunks"],
            entities=retrieval_results.get("entities", []),
            relationships=retrieval_results.get("relationships", [])
        )

        # Prepare response
        response = QueryResponse(
            query=answer.query,
            answer=answer.answer,
            context_chunks=answer.context_chunks,
            entities=answer.entities,
            relationships=answer.relationships,
            model_used=answer.model_used,
            generation_time=answer.generation_time,
            retrieval_stats=retriever.get_retrieval_stats()
        )

        logger.info(f"Query processed successfully in {answer.generation_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    file_path: Optional[str] = Form(None),
    directory_path: Optional[str] = Form(None)
):
    """
    Ingest documents into the RAG system.

    This endpoint processes documents (PDF, Markdown, text files) and
    extracts entities, relationships, and generates embeddings for storage.
    """
    try:
        if file_path:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

            logger.info(f"Ingesting file: {file_path}")
            stats = ingester.ingest_file(file_path)

        elif directory_path:
            if not os.path.exists(directory_path):
                raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")

            logger.info(f"Ingesting directory: {directory_path}")
            stats = ingester.ingest_directory(directory_path)

        else:
            raise HTTPException(status_code=400, detail="Either file_path or directory_path must be provided")

        message = f"Successfully ingested {stats.get('chunks', 0)} chunks, {stats.get('entities', 0)} entities, {stats.get('relationships', 0)} relationships"

        response = IngestResponse(
            success=True,
            stats=stats,
            message=message
        )

        logger.info(f"Ingestion completed: {message}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and status."""
    try:
        stats = ingester.get_ingestion_stats()

        return SystemStats(
            database_stats=stats["database_stats"],
            embedding_dimension=stats["embedding_dimension"],
            embedding_model=stats["embedding_model"],
            llm_model=stats["llm_model"],
            api_status="healthy"
        )

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system stats: {str(e)}")

@app.delete("/clear")
async def clear_system():
    """Clear all data from the system."""
    try:
        ingester.clear_all_data()

        return {
            "success": True,
            "message": "All system data cleared successfully"
        }

    except Exception as e:
        logger.error(f"Error clearing system: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing system: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2025-01-16T09:10:00Z",  # Current timestamp would be dynamic in real implementation
        "version": "1.0.0"
    }

@app.get("/config")
async def get_configuration():
    """Get current system configuration."""
    try:
        return {
            "database": {
                "path": config.database.path,
                "vector_dim": config.database.vector_dim
            },
            "embeddings": {
                "model_name": config.embeddings.model_name,
                "batch_size": config.embeddings.batch_size
            },
            "llm": {
                "base_url": config.llm.base_url,
                "model_name": config.llm.model_name,
                "temperature": config.llm.temperature
            },
            "retrieval": {
                "vector_top_k": config.retrieval.vector_top_k,
                "graph_depth": config.retrieval.graph_depth,
                "hybrid_alpha": config.retrieval.hybrid_alpha
            }
        }

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

# Optional: File upload endpoint for document ingestion
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document file.

    This endpoint accepts file uploads and automatically ingests them
    into the RAG system.
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".md", ".txt", ".docx", ".xlsx", ".pptx"}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, f"upload_{file.filename}")
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Ingest the file
        stats = ingester.ingest_file(temp_path)

        # Clean up temporary file
        os.remove(temp_path)

        return IngestResponse(
            success=True,
            stats=stats,
            message=f"Successfully ingested uploaded file '{file.filename}'"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

# Development endpoint for testing
@app.post("/test-retrieval")
async def test_retrieval(request: QueryRequest):
    """Test retrieval without answer generation (for debugging)."""
    try:
        # Perform retrieval only
        retrieval_results = retriever.retrieve_with_entities(request.query, request.top_k)

        return {
            "query": request.query,
            "retrieval_results": retrieval_results,
            "retrieval_stats": retriever.get_retrieval_stats()
        }

    except Exception as e:
        logger.error(f"Error in test retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Error in test retrieval: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers
    )
