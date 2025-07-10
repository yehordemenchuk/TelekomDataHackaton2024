import asyncio
import hashlib
import os
import shutil
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import aiofiles
import magic
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredRTFLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..models.schemas import DocumentType
from .langchain_config import langchain_config

logger = logging.getLogger(__name__)

class LangChainDocumentProcessor:
    SUPPORTED_FORMATS = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain',
        'rtf': 'application/rtf'
    }
    
    def __init__(self):
        self.storage_path = Path(settings.document_storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.config = langchain_config
        self.text_splitter = self.config.text_splitter
    
    async def process_uploaded_document(
        self,
        file_content: bytes,
        filename: str,
        title: str,
        document_type: DocumentType,
        owner_id: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], str]:
        try:
            file_format = await self._detect_file_format(file_content, filename)
            if not file_format:
                raise ValueError(f"Unsupported file format: {filename}")
            
            file_path = await self._save_file(file_content, filename, owner_id)
            
            content = await self._extract_content_with_langchain(file_path, file_format)
            
            extracted_metadata = await self.extract_document_metadata(file_content, file_format)
            
            final_metadata = metadata or {}
            final_metadata.update(extracted_metadata)
            
            document_info = {
                "title": title,
                "filename": filename,
                "file_path": str(file_path),
                "document_type": document_type.value,
                "owner_id": owner_id,
                "metadata": final_metadata,
                "file_size": len(file_content)
            }
            
            logger.info(f"Processed document: {filename} using LangChain")
            return document_info, content
            
        except Exception as e:
            logger.error(f"Error processing document {filename} with LangChain: {str(e)}")
            raise
    
    async def _detect_file_format(self, file_content: bytes, filename: str) -> Optional[str]:
        try:
            mime_type = magic.from_buffer(file_content, mime=True)
            
            for format_name, format_mime in self.SUPPORTED_FORMATS.items():
                if mime_type == format_mime:
                    return format_name
            
            file_ext = Path(filename).suffix.lower().lstrip('.')
            if file_ext in self.SUPPORTED_FORMATS:
                return file_ext
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting file format: {str(e)}")
            return None
    
    async def _save_file(self, file_content: bytes, filename: str, owner_id: int) -> Path:
        try:
            user_dir = self.storage_path / f"user_{owner_id}"
            user_dir.mkdir(exist_ok=True)
            
            timestamp = int(asyncio.get_event_loop().time())
            file_extension = Path(filename).suffix
            unique_filename = f"{timestamp}_{filename}"
            
            file_path = user_dir / unique_filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            raise
    
    async def _extract_content_with_langchain(self, file_path: Path, file_format: str) -> str:
        try:
            if file_format == 'pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_format == 'docx':
                loader = Docx2txtLoader(str(file_path))
            elif file_format == 'txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_format == 'rtf':
                loader = UnstructuredRTFLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported format for LangChain extraction: {file_format}")
            
            documents = loader.load()
            
            content = "\n\n".join([doc.page_content for doc in documents])
            
            logger.info(f"Extracted content using LangChain {file_format} loader")
            return content.strip()
                
        except Exception as e:
            logger.error(f"Error extracting content from {file_format} with LangChain: {str(e)}")
            raise
    
    async def load_document_as_langchain_docs(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        try:
            file_path_obj = Path(file_path)
            file_format = file_path_obj.suffix.lower().lstrip('.')
            
            if file_format == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_format == 'docx':
                loader = Docx2txtLoader(file_path)
            elif file_format == 'txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_format == 'rtf':
                loader = UnstructuredRTFLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            documents = loader.load()
            
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            logger.info(f"Loaded {len(documents)} document chunks using LangChain")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document as LangChain docs: {str(e)}")
            raise
    
    async def split_document_into_chunks(
        self,
        content: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[str]:
        try:
            if chunk_size or chunk_overlap:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size or settings.max_chunk_size,
                    chunk_overlap=chunk_overlap or settings.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
            else:
                text_splitter = self.text_splitter
            
            chunks = text_splitter.split_text(content)
            
            logger.info(f"Split document into {len(chunks)} chunks using LangChain")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting document into chunks with LangChain: {str(e)}")
            raise
    
    async def process_document_with_chunks(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        try:
            documents = await self.load_document_as_langchain_docs(file_path, metadata)
            
            chunked_documents = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_page": chunk_metadata.get("page", 0)
                    })
                    
                    chunked_documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
            
            logger.info(f"Processed document into {len(chunked_documents)} chunks")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error processing document with chunks: {str(e)}")
            raise
    
    async def extract_document_metadata(self, file_content: bytes, file_format: str) -> Dict[str, Any]:
        try:
            metadata = {
                "file_format": file_format,
                "file_size": len(file_content),
                "processed_at": asyncio.get_event_loop().time(),
                "processor": "LangChain"
            }
            
            if file_format == 'pdf':
                metadata.update(await self._extract_pdf_metadata_langchain(file_content))
            elif file_format == 'docx':
                metadata.update(await self._extract_docx_metadata_langchain(file_content))
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata with LangChain: {str(e)}")
            return {"file_format": file_format, "processor": "LangChain"}
    
    async def _extract_pdf_metadata_langchain(self, file_content: bytes) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                
                return {
                    "total_pages": len(documents),
                    "page_count": len(documents),
                    "extraction_method": "LangChain_PyPDFLoader"
                }
                
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error extracting PDF metadata with LangChain: {str(e)}")
            return {"extraction_method": "LangChain_PyPDFLoader", "error": str(e)}
    
    async def _extract_docx_metadata_langchain(self, file_content: bytes) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                loader = Docx2txtLoader(temp_file_path)
                documents = loader.load()
                
                total_content = "\n".join([doc.page_content for doc in documents])
                
                return {
                    "word_count": len(total_content.split()),
                    "character_count": len(total_content),
                    "extraction_method": "LangChain_Docx2txtLoader"
                }
                
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error extracting DOCX metadata with LangChain: {str(e)}")
            return {"extraction_method": "LangChain_Docx2txtLoader", "error": str(e)}
    
    async def batch_process_documents(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        try:
            all_documents = []
            
            for file_path in file_paths:
                try:
                    documents = await self.process_document_with_chunks(file_path, metadata)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
            
            logger.info(f"Batch processed {len(file_paths)} files into {len(all_documents)} chunks")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error in batch processing with LangChain: {str(e)}")
            raise
    
    async def validate_document_structure(
        self,
        documents: List[Document],
        required_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        try:
            validation_results = {
                "total_documents": len(documents),
                "total_content_length": sum(len(doc.page_content) for doc in documents),
                "average_chunk_size": 0,
                "sections_found": [],
                "missing_sections": [],
                "validation_score": 0.0
            }
            
            if documents:
                validation_results["average_chunk_size"] = validation_results["total_content_length"] / len(documents)
            
            all_content = "\n".join([doc.page_content for doc in documents])
            
            if required_sections:
                sections_found = []
                for section in required_sections:
                    if section.lower() in all_content.lower():
                        sections_found.append(section)
                
                validation_results["sections_found"] = sections_found
                validation_results["missing_sections"] = [s for s in required_sections if s not in sections_found]
                validation_results["validation_score"] = len(sections_found) / len(required_sections)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating document structure: {str(e)}")
            raise
    
    async def delete_document_file(self, file_path: str):
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                path_obj.unlink()
                logger.info(f"Deleted document file: {file_path}")
            else:
                logger.warning(f"File not found for deletion: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting document file {file_path}: {str(e)}")
    
    async def validate_document_content(self, content: str, document_type: DocumentType) -> Dict[str, Any]:
        try:
            validation_results = {
                "content_length": len(content),
                "word_count": len(content.split()),
                "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
                "has_content": len(content.strip()) > 0,
                "document_type": document_type.value,
                "validation_passed": True,
                "issues": [],
                "recommendations": []
            }
            
            if validation_results["content_length"] < 50:
                validation_results["issues"].append("Document content is very short")
                validation_results["validation_passed"] = False
            
            if validation_results["word_count"] < 10:
                validation_results["issues"].append("Document has very few words")
                validation_results["validation_passed"] = False
            
            required_elements = {
                DocumentType.CONTRACT: ["party", "agreement", "term"],
                DocumentType.LEASE: ["tenant", "landlord", "rent", "property"],
                DocumentType.NDA: ["confidential", "disclosure", "information"],
                DocumentType.WILL: ["testator", "executor", "beneficiary"],
                DocumentType.POWER_OF_ATTORNEY: ["attorney", "principal", "power"]
            }
            
            if document_type in required_elements:
                found_elements = []
                for element in required_elements[document_type]:
                    if element.lower() in content.lower():
                        found_elements.append(element)
                
                missing_elements = [e for e in required_elements[document_type] if e not in found_elements]
                
                if missing_elements:
                    validation_results["issues"].append(f"Missing key elements: {', '.join(missing_elements)}")
                    validation_results["recommendations"].append(f"Consider adding content related to: {', '.join(missing_elements)}")
            
            logger.info(f"Validated document content for {document_type.value}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating document content: {str(e)}")
            raise
    
    async def get_storage_statistics(self, owner_id: Optional[int] = None) -> Dict[str, Any]:
        try:
            if owner_id:
                user_dir = self.storage_path / f"user_{owner_id}"
                if not user_dir.exists():
                    return {"total_files": 0, "total_size": 0, "storage_path": str(user_dir)}
                
                files = list(user_dir.glob("*"))
            else:
                files = list(self.storage_path.rglob("*"))
                files = [f for f in files if f.is_file()]
            
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            file_types = {}
            for file in files:
                if file.is_file():
                    ext = file.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                "total_files": len(files),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "storage_path": str(self.storage_path),
                "processor": "LangChain"
            }
            
        except Exception as e:
            logger.error(f"Error getting storage statistics: {str(e)}")
            raise 