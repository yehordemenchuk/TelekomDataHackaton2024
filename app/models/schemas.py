from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    CONTRACT = "contract"
    AGREEMENT = "agreement"
    DECLARATION = "declaration"
    MOTION = "motion"
    BRIEF = "brief"
    MEMORANDUM = "memorandum"
    WILL = "will"
    POWER_OF_ATTORNEY = "power_of_attorney"
    LEASE = "lease"
    NDA = "nda"

# User Schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Document Schemas
class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    document_type: DocumentType
    metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    document_id: str  # Vector DB document ID
    title: str
    filename: str
    document_type: str
    metadata: Optional[Dict[str, Any]]
    owner_id: int
    created_at: datetime
    file_size: Optional[int] = None

class DocumentSearch(BaseModel):
    query: str = Field(..., min_length=1)
    document_types: Optional[List[DocumentType]] = None
    limit: int = Field(default=10, le=50)
    include_content: bool = False

class SearchResult(BaseModel):
    document_id: str  # Vector DB document ID
    title: str
    document_type: str
    relevance_score: float
    content_snippet: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float

# Document Generation Schemas
class DocumentGenerationRequest(BaseModel):
    document_type: DocumentType
    prompt: str = Field(..., min_length=10)
    template_id: Optional[int] = None
    context_document_ids: Optional[List[str]] = None  # Vector DB document IDs
    variables: Optional[Dict[str, Any]] = None
    generation_params: Optional[Dict[str, Any]] = None

class DocumentGenerationResponse(BaseModel):
    id: int
    document_type: str
    generated_content: str
    template_used: Optional[str]
    context_documents: Optional[List[str]]  # Vector DB document IDs
    created_at: datetime
    
    class Config:
        from_attributes = True

# Template Schemas
class TemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    document_type: DocumentType
    template_content: str = Field(..., min_length=1)
    variables: Optional[Dict[str, Any]] = None

class TemplateResponse(BaseModel):
    id: int
    name: str
    document_type: str
    template_content: str
    variables: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# RAG Schemas
class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1)
    document_types: Optional[List[DocumentType]] = None
    max_chunks: int = Field(default=5, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class RAGContext(BaseModel):
    chunks: List[str]
    source_documents: List[str]  # Vector DB document IDs
    relevance_scores: List[float]

# MCP Schemas
class MCPToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class MCPResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

# Legal Analysis Schemas
class LegalAnalysisRequest(BaseModel):
    document_content: str = Field(..., min_length=1)
    analysis_type: str = Field(..., regex="^(summary|risk_assessment|compliance_check|clause_extraction)$")
    context_documents: Optional[List[str]] = None  # Vector DB document IDs

class LegalAnalysisResponse(BaseModel):
    analysis_type: str
    results: Dict[str, Any]
    confidence: float
    context_used: List[str]  # Vector DB document IDs
    generated_at: datetime

# Error Schemas
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Health Check Schema
class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str] 