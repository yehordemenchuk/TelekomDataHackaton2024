import faiss
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import logging
import uuid
import pickle
from datetime import datetime
from ..config import settings
import numpy as np
import re

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.faiss_index_path = Path(settings.faiss_index_path)
        self.metadata_storage_path = Path(settings.metadata_storage_path)
        
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.metadata_store = {}
        self.id_to_index_map = {}
        self.index_to_id_map = {}
        self.next_index = 0
        
        self._load_index()
        self._load_metadata()
    
    def _load_index(self):
        index_file = self.faiss_index_path / "legal_documents.index"
        map_file = self.faiss_index_path / "id_mappings.pkl"
        
        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
                if map_file.exists():
                    with open(map_file, 'rb') as f:
                        mappings = pickle.load(f)
                        self.id_to_index_map = mappings.get('id_to_index', {})
                        self.index_to_id_map = mappings.get('index_to_id', {})
                        self.next_index = mappings.get('next_index', 0)
                        
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}, creating new one")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata_store = {}
        self.id_to_index_map = {}
        self.index_to_id_map = {}
        self.next_index = 0
        logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def _save_index(self):
        try:
            index_file = self.faiss_index_path / "legal_documents.index"
            map_file = self.faiss_index_path / "id_mappings.pkl"
            
            faiss.write_index(self.index, str(index_file))
            
            mappings = {
                'id_to_index': self.id_to_index_map,
                'index_to_id': self.index_to_id_map,
                'next_index': self.next_index
            }
            with open(map_file, 'wb') as f:
                pickle.dump(mappings, f)
                
            logger.info("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _load_metadata(self):
        try:
            for metadata_file in self.metadata_storage_path.glob("*.json"):
                doc_id = metadata_file.stem
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata_store[doc_id] = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata_store)} documents")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self.metadata_store = {}
    
    def _save_metadata(self, document_id: str, metadata: Dict[str, Any]):
        try:
            metadata_file = self.metadata_storage_path / f"{document_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            self.metadata_store[document_id] = metadata
        except Exception as e:
            logger.error(f"Failed to save metadata for {document_id}: {e}")
    
    def _delete_metadata(self, document_id: str):
        try:
            metadata_file = self.metadata_storage_path / f"{document_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            if document_id in self.metadata_store:
                del self.metadata_store[document_id]
        except Exception as e:
            logger.error(f"Failed to delete metadata for {document_id}: {e}")

    async def index_document(
        self,
        title: str,
        filename: str,
        file_path: str,
        document_type: str,
        content: str,
        owner_id: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        try:
            document_id = str(uuid.uuid4())
            
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            existing_doc_id = await self.check_document_exists(content_hash, owner_id)
            if existing_doc_id:
                logger.info(f"Document already exists with hash: {content_hash}")
                return existing_doc_id
            
            chunks = self._chunk_text(content)
            
            embeddings = self.embedding_model.encode(chunks)
            faiss.normalize_L2(embeddings)
            
            doc_metadata = metadata or {}
            doc_metadata.update({
                "document_id": document_id,
                "document_type": document_type,
                "title": title,
                "filename": filename,
                "file_path": file_path,
                "owner_id": owner_id,
                "content_hash": content_hash,
                "created_at": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "chunks": [{"index": i, "content": chunk} for i, chunk in enumerate(chunks)]
            })
            
            start_index = self.next_index
            self.index.add(embeddings)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"doc_{document_id}_chunk_{i}"
                index_pos = start_index + i
                self.id_to_index_map[chunk_id] = index_pos
                self.index_to_id_map[index_pos] = chunk_id
            
            self.next_index += len(chunks)
            
            self._save_metadata(document_id, doc_metadata)
            
            self._save_index()
            
            logger.info(f"Indexed document {document_id} with {len(chunks)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise

    async def search_documents(
        self,
        query: str,
        owner_id: int,
        document_type: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.info("No documents indexed yet")
                return []
            
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, min(limit * 3, self.index.ntotal))
            
            search_results = []
            seen_documents = set()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                    
                chunk_id = self.index_to_id_map.get(idx)
                if not chunk_id:
                    continue
                    
                document_id = chunk_id.split("_chunk_")[0].replace("doc_", "")
                
                if document_id not in self.metadata_store:
                    continue
                    
                doc_metadata = self.metadata_store[document_id]
                
                if doc_metadata.get("owner_id") != owner_id:
                    continue
                    
                if document_type and doc_metadata.get("document_type") != document_type:
                    continue
                
                similarity_score = float(score)
                
                if similarity_score >= min_score:
                    chunk_index = int(chunk_id.split("_chunk_")[1])
                    chunk_content = ""
                    
                    if "chunks" in doc_metadata:
                        for chunk in doc_metadata["chunks"]:
                            if chunk["index"] == chunk_index:
                                chunk_content = chunk["content"]
                                break
                    
                    search_results.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "title": doc_metadata.get("title", ""),
                        "document_type": doc_metadata.get("document_type", ""),
                        "filename": doc_metadata.get("filename", ""),
                        "chunk_content": chunk_content,
                        "chunk_index": chunk_index,
                        "similarity_score": similarity_score,
                        "metadata": doc_metadata
                    })
                    
                    seen_documents.add(document_id)
                    
                    if len(search_results) >= limit:
                        break
            
            search_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"Found {len(search_results)} relevant chunks for query")
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    async def search_similar_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.info("No documents indexed yet")
                return []
            
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, min(max_results * 2, self.index.ntotal))
            
            search_results = []
            
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                    
                chunk_id = self.index_to_id_map.get(idx)
                if not chunk_id:
                    continue
                    
                document_id = chunk_id.split("_chunk_")[0].replace("doc_", "")
                
                if document_id not in self.metadata_store:
                    continue
                    
                doc_metadata = self.metadata_store[document_id]
                
                if document_types and doc_metadata.get("document_type") not in document_types:
                    continue
                
                similarity_score = float(score)
                
                if similarity_score >= similarity_threshold:
                    chunk_index = int(chunk_id.split("_chunk_")[1])
                    chunk_content = ""
                    
                    if "chunks" in doc_metadata:
                        for chunk in doc_metadata["chunks"]:
                            if chunk["index"] == chunk_index:
                                chunk_content = chunk["content"]
                                break
                    
                    search_results.append({
                        "chunk_id": chunk_id,
                        "content": chunk_content,
                        "metadata": doc_metadata,
                        "similarity_score": similarity_score
                    })
                    
                    if len(search_results) >= max_results:
                        break
            
            search_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return search_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    async def get_relevant_context(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        max_chunks: int = 5,
        similarity_threshold: float = 0.7
    ) -> Tuple[List[str], List[str], List[float]]:
        try:
            results = await self.search_similar_documents(
                query=query,
                document_types=document_types,
                max_results=max_chunks,
                similarity_threshold=similarity_threshold
            )
            
            chunks = [result["content"] for result in results]
            document_ids = [result["metadata"]["document_id"] for result in results]
            scores = [result["similarity_score"] for result in results]
            
            return chunks, document_ids, scores
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            raise

    async def update_document_index(
        self,
        document_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        try:
            if document_id not in self.metadata_store:
                raise ValueError(f"Document with ID {document_id} not found")
            
            original_metadata = self.metadata_store[document_id]
            
            await self.delete_document_index(document_id)
            
            updated_doc_id = await self.index_document(
                title=original_metadata.get("title", "Updated Document"),
                filename=original_metadata.get("filename", "updated_file"),
                file_path=original_metadata.get("file_path", ""),
                document_type=original_metadata.get("document_type", "document"),
                content=new_content,
                owner_id=original_metadata.get("owner_id"),
                metadata=metadata
            )
            
            logger.info(f"Updated index for document {document_id}")
            return updated_doc_id
            
        except Exception as e:
            logger.error(f"Error updating document index {document_id}: {str(e)}")
            raise

    async def delete_document_index(self, document_id: str):
        try:
            if document_id not in self.metadata_store:
                logger.warning(f"Document {document_id} not found in metadata store")
                return
            
            doc_metadata = self.metadata_store[document_id]
            total_chunks = doc_metadata.get("total_chunks", 0)
            
            chunk_ids_to_remove = []
            indices_to_remove = []
            
            for chunk_index in range(total_chunks):
                chunk_id = f"doc_{document_id}_chunk_{chunk_index}"
                if chunk_id in self.id_to_index_map:
                    faiss_index = self.id_to_index_map[chunk_id]
                    chunk_ids_to_remove.append(chunk_id)
                    indices_to_remove.append(faiss_index)
            
            for chunk_id in chunk_ids_to_remove:
                faiss_index = self.id_to_index_map.pop(chunk_id, None)
                if faiss_index is not None:
                    self.index_to_id_map.pop(faiss_index, None)
            
            self._delete_metadata(document_id)
            
            self._save_index()
            
            logger.info(f"Deleted {len(chunk_ids_to_remove)} chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document index {document_id}: {str(e)}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= settings.max_chunk_size:
                current_chunk += (sentence + " ")
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        if settings.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_chunk_overlap(chunks)
        
        return chunks

    def _add_chunk_overlap(self, chunks: List[str]) -> List[str]:
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                previous_chunk = chunks[i - 1]
                overlap_words = previous_chunk.split()[-settings.chunk_overlap:]
                overlap_text = " ".join(overlap_words)
                
                overlapped_chunk = f"{overlap_text} {chunk}"
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks

    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            total_chunks = self.index.ntotal if self.index else 0
            
            doc_types = {}
            owners = set()
            total_documents = len(self.metadata_store)
            
            for doc_id, metadata in self.metadata_store.items():
                doc_type = metadata.get("document_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                owners.add(metadata.get("owner_id"))
            
            return {
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "total_owners": len(owners),
                "document_types": doc_types,
                "embedding_model": settings.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    async def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        try:
            search_results = []
            
            for doc_id, metadata in self.metadata_store.items():
                matches = True
                for key, value in metadata_filter.items():
                    if key not in metadata or metadata[key] != value:
                        matches = False
                        break
                
                if matches:
                    chunks = metadata.get("chunks", [])
                    for chunk in chunks:
                        chunk_id = f"doc_{doc_id}_chunk_{chunk['index']}"
                        search_results.append({
                            "chunk_id": chunk_id,
                            "content": chunk["content"],
                            "metadata": metadata
                        })
                        
                        if len(search_results) >= max_results:
                            break
                
                if len(search_results) >= max_results:
                    break
            
            return search_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            raise

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            if document_id not in self.metadata_store:
                return None
            
            metadata = self.metadata_store[document_id]
            return {
                "document_id": document_id,
                "title": metadata.get("title"),
                "filename": metadata.get("filename"),
                "document_type": metadata.get("document_type"),
                "owner_id": metadata.get("owner_id"),
                "created_at": metadata.get("created_at"),
                "total_chunks": metadata.get("total_chunks"),
                "metadata": {k: v for k, v in metadata.items() if k not in [
                    "document_id", "title", "filename", "document_type", 
                    "owner_id", "created_at", "total_chunks", "chunks", "content_hash", "file_path"
                ]}
            }
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            raise

    async def list_user_documents(
        self, 
        owner_id: int, 
        document_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        try:
            documents = []
            
            for doc_id, metadata in self.metadata_store.items():
                if metadata.get("owner_id") != owner_id:
                    continue
                
                if document_types and metadata.get("document_type") not in document_types:
                    continue
                
                documents.append({
                    "document_id": doc_id,
                    "title": metadata.get("title"),
                    "filename": metadata.get("filename"),
                    "document_type": metadata.get("document_type"),
                    "owner_id": metadata.get("owner_id"),
                    "created_at": metadata.get("created_at"),
                    "total_chunks": metadata.get("total_chunks"),
                    "metadata": {k: v for k, v in metadata.items() if k not in [
                        "document_id", "title", "filename", "document_type", 
                        "owner_id", "created_at", "total_chunks", "chunks", "content_hash", "file_path"
                    ]}
                })
                
                if len(documents) >= limit:
                    break
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents for user {owner_id}: {str(e)}")
            raise

    async def get_document_content(self, document_id: str) -> str:
        try:
            if document_id not in self.metadata_store:
                raise ValueError(f"Document {document_id} not found")
            
            metadata = self.metadata_store[document_id]
            chunks = metadata.get("chunks", [])
            
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")
            
            chunks.sort(key=lambda x: x["index"])
            
            content = " ".join([chunk["content"] for chunk in chunks])
            return content
            
        except Exception as e:
            logger.error(f"Error getting content for document {document_id}: {str(e)}")
            raise

    async def check_document_exists(self, content_hash: str, owner_id: int) -> Optional[str]:
        try:
            for doc_id, metadata in self.metadata_store.items():
                if (metadata.get("content_hash") == content_hash and 
                    metadata.get("owner_id") == owner_id):
                    return doc_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            raise

    async def get_document_statistics(self, owner_id: Optional[int] = None) -> Dict[str, Any]:
        try:
            document_types = {}
            total_documents = 0
            total_chunks = 0
            
            for doc_id, metadata in self.metadata_store.items():
                if owner_id and metadata.get("owner_id") != owner_id:
                    continue
                
                doc_type = metadata.get("document_type", "unknown")
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
                total_documents += 1
                total_chunks += metadata.get("total_chunks", 0)
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "document_types": document_types,
                "embedding_model": settings.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error getting document statistics: {str(e)}")
            raise 