from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

# Import configuration and database
from .config import settings
from .database import init_db

# Import routers
from .routers import auth

# Import services for initialization
from .services.rag_service import RAGService
from .services.claude_client import ClaudeClient
from .services.mcp_service import MCPService
from .services.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global service instances
rag_service = None
claude_client = None
mcp_service = None
document_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global rag_service, claude_client, mcp_service, document_processor
    
    # Startup
    logger.info("Starting Legal AI application...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Initialize services
        rag_service = RAGService()
        claude_client = ClaudeClient()
        mcp_service = MCPService()
        document_processor = DocumentProcessor()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal AI application...")

# Create FastAPI application
app = FastAPI(
    title="Legal AI System",
    description="AI-powered legal document processing and generation system with RAG and MCP integration",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["yourdomain.com", "*.yourdomain.com"]
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

# Include routers
app.include_router(auth.router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        from .database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    # Check RAG service
    try:
        if rag_service:
            stats = await rag_service.get_collection_stats()
            rag_status = "healthy"
        else:
            rag_status = "not_initialized"
    except Exception:
        rag_status = "unhealthy"
    
    # Check MCP service
    try:
        if mcp_service:
            tools = mcp_service.get_available_tools()
            mcp_status = "healthy"
        else:
            mcp_status = "not_initialized"
    except Exception:
        mcp_status = "unhealthy"
    
    overall_status = "healthy" if all(
        status == "healthy" for status in [db_status, rag_status, mcp_status]
    ) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": {
            "database": db_status,
            "rag": rag_status,
            "mcp": mcp_status,
            "claude": "healthy" if claude_client else "not_initialized"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Legal AI System API",
        "version": "1.0.0",
        "docs": "/docs" if settings.debug else "Documentation not available in production",
        "health": "/health"
    }

# API Info endpoint
@app.get("/api/v1/info")
async def api_info():
    """Get API information."""
    return {
        "name": "Legal AI System API",
        "version": "1.0.0",
        "description": "AI-powered legal document processing and generation system",
        "features": [
            "Document upload and processing",
            "RAG-based document search",
            "AI-powered document generation",
            "Legal analysis and validation",
            "MCP integration for external tools",
            "User authentication and authorization"
        ],
        "supported_document_types": [
            "contract", "agreement", "declaration", "motion", 
            "brief", "memorandum", "will", "power_of_attorney", 
            "lease", "nda"
        ],
        "supported_file_formats": ["pdf", "docx", "txt", "rtf"]
    }

# Service status endpoint
@app.get("/api/v1/status")
async def service_status():
    """Get detailed service status."""
    try:
        status_info = {
            "application": {
                "name": "Legal AI System",
                "version": "1.0.0",
                "uptime": time.time(),
                "debug_mode": settings.debug
            },
            "services": {},
            "configuration": {
                "max_chunk_size": settings.max_chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "embedding_model": settings.embedding_model,
                "claude_model": settings.claude_model
            }
        }
        
        # RAG service status
        if rag_service:
            try:
                rag_stats = await rag_service.get_collection_stats()
                status_info["services"]["rag"] = {
                    "status": "healthy",
                    "stats": rag_stats
                }
            except Exception as e:
                status_info["services"]["rag"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            status_info["services"]["rag"] = {"status": "not_initialized"}
        
        # MCP service status
        if mcp_service:
            try:
                tools = mcp_service.get_available_tools()
                status_info["services"]["mcp"] = {
                    "status": "healthy",
                    "available_tools": len(tools),
                    "tools": [tool["name"] for tool in tools]
                }
            except Exception as e:
                status_info["services"]["mcp"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            status_info["services"]["mcp"] = {"status": "not_initialized"}
        
        # Claude service status
        status_info["services"]["claude"] = {
            "status": "healthy" if claude_client else "not_initialized",
            "model": settings.claude_model if claude_client else None
        }
        
        # Document processor status
        status_info["services"]["document_processor"] = {
            "status": "healthy" if document_processor else "not_initialized",
            "supported_formats": list(document_processor.SUPPORTED_FORMATS.keys()) if document_processor else []
        }
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting service status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get service status")

# Utility functions for accessing services in route handlers
def get_rag_service() -> RAGService:
    """Get RAG service instance."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    return rag_service

def get_claude_client() -> ClaudeClient:
    """Get Claude client instance."""
    if not claude_client:
        raise HTTPException(status_code=503, detail="Claude service not available")
    return claude_client

def get_mcp_service() -> MCPService:
    """Get MCP service instance."""
    if not mcp_service:
        raise HTTPException(status_code=503, detail="MCP service not available")
    return mcp_service

def get_document_processor() -> DocumentProcessor:
    """Get document processor instance."""
    if not document_processor:
        raise HTTPException(status_code=503, detail="Document processor not available")
    return document_processor

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    ) 