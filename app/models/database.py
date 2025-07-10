from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    generations = relationship("DocumentGeneration", back_populates="user")

class DocumentGeneration(Base):
    __tablename__ = "document_generations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    document_type = Column(String(50), nullable=False)
    prompt = Column(Text, nullable=False)
    generated_content = Column(Text, nullable=False)
    template_used = Column(String(100))
    context_documents = Column(JSON)  # Vector DB document IDs used as context
    generation_metadata = Column(JSON)  # Model params, etc.
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="generations")

class Template(Base):
    __tablename__ = "templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    document_type = Column(String(50), nullable=False)
    template_content = Column(Text, nullable=False)
    variables = Column(JSON)  # Template variables schema
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class SearchQuery(Base):
    __tablename__ = "search_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    query = Column(Text, nullable=False)
    results_count = Column(Integer, default=0)
    search_metadata = Column(JSON)  # Search parameters, filters, etc.
    result_document_ids = Column(JSON)  # Vector DB document IDs from search results
    created_at = Column(DateTime, default=func.now()) 