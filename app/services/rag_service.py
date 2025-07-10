import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import logging
import uuid
from datetime import datetime
from ..config import settings
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .langchain_config import langchain_config, LegalRAGChain

logger = logging.getLogger(__name__)

class LangChainRAGService:
    def __init__(self):
        self.config = langchain_config
        self.embedding_model = self.config.embedding_model
        self.text_splitter = self.config.text_splitter
        self.llm = self.config.anthropic_llm
        
        self.vector_store_path = Path(settings.faiss_index_path)
        self.metadata_storage_path = Path(settings.metadata_storage_path)
        self.vector_store_path.mkdir(exist_ok=True)
        self.metadata_storage_path.mkdir(exist_ok=True)
        
        self.vector_store = None
        self.metadata_store = {}
        
        self._load_vector_store()
        self._load_metadata()
    
    def _load_vector_store(self):
        index_path = self.vector_store_path / "faiss_index"
        if index_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(index_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded FAISS vector store with {self.vector_store.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}, creating new one")
                self._create_new_vector_store()
        else:
            self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        dummy_doc = Document(page_content="dummy", metadata={"source": "init"})
        self.vector_store = FAISS.from_documents([dummy_doc], self.embedding_model)
        self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
        logger.info("Created new FAISS vector store")
    
    def _save_vector_store(self):
        try:
            index_path = self.vector_store_path / "faiss_index"
            self.vector_store.save_local(str(index_path))
            logger.info("FAISS vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
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
            
            text_chunks = self.text_splitter.split_text(content)
            
            documents = []
            for i, chunk in enumerate(text_chunks):
                doc_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "document_type": document_type,
                    "title": title,
                    "filename": filename,
                    "file_path": file_path,
                    "owner_id": owner_id,
                    "content_hash": content_hash,
                    "created_at": datetime.now().isoformat(),
                    "source": f"doc_{document_id}_chunk_{i}"
                }
                
                if metadata:
                    doc_metadata.update(metadata)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            if self.vector_store.index.ntotal == 0:
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            else:
                self.vector_store.add_documents(documents)
            
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
                "total_chunks": len(text_chunks),
                "chunks": [{"index": i, "content": chunk} for i, chunk in enumerate(text_chunks)]
            })
            
            self._save_metadata(document_id, doc_metadata)
            self._save_vector_store()
            
            logger.info(f"Indexed document {document_id} with {len(text_chunks)} chunks using LangChain")
            return document_id
            
        except Exception as e:
            logger.error(f"Error indexing document with LangChain: {str(e)}")
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
            if not self.vector_store or self.vector_store.index.ntotal == 0:
                logger.info("No documents indexed yet")
                return []
            
            filter_dict = {"owner_id": owner_id}
            if document_type:
                filter_dict["document_type"] = document_type
            
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=limit * 2,
                filter=filter_dict
            )
            
            search_results = []
            seen_documents = set()
            
            for doc, score in docs_and_scores:
                similarity_score = 1 - score
                
                if similarity_score >= min_score:
                    doc_metadata = doc.metadata
                    document_id = doc_metadata.get("document_id")
                    
                    if document_id not in seen_documents:
                        search_results.append({
                            "chunk_id": doc_metadata.get("source", ""),
                            "document_id": document_id,
                            "title": doc_metadata.get("title", ""),
                            "document_type": doc_metadata.get("document_type", ""),
                            "filename": doc_metadata.get("filename", ""),
                            "chunk_content": doc.page_content,
                            "chunk_index": doc_metadata.get("chunk_index", 0),
                            "similarity_score": similarity_score,
                            "metadata": doc_metadata
                        })
                        
                        seen_documents.add(document_id)
                        
                        if len(search_results) >= limit:
                            break
            
            search_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"Found {len(search_results)} relevant chunks for query using LangChain")
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching documents with LangChain: {str(e)}")
            return []

    async def search_similar_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        try:
            if not self.vector_store or self.vector_store.index.ntotal == 0:
                logger.info("No documents indexed yet")
                return []
            
            filter_dict = {}
            if document_types:
                filter_dict["document_type"] = {"$in": document_types}
            
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=max_results * 2,
                filter=filter_dict if filter_dict else None
            )
            
            search_results = []
            
            for doc, score in docs_and_scores:
                similarity_score = 1 - score
                
                if similarity_score >= similarity_threshold:
                    search_results.append({
                        "chunk_id": doc.metadata.get("source", ""),
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": similarity_score
                    })
                    
                    if len(search_results) >= max_results:
                        break
            
            search_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return search_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching documents with LangChain: {str(e)}")
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
            logger.error(f"Error getting relevant context with LangChain: {str(e)}")
            raise

    async def create_rag_qa_chain(self) -> RetrievalQA:
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            rag_chain = LegalRAGChain(self.config, self.vector_store)
            return rag_chain.create_qa_chain()
            
        except Exception as e:
            logger.error(f"Error creating RAG QA chain: {str(e)}")
            raise

    async def ask_question(
        self,
        question: str,
        owner_id: Optional[int] = None,
        document_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        try:
            if not self.vector_store:
                return {"answer": "No documents indexed yet", "sources": []}
            
            filter_dict = {}
            if owner_id:
                filter_dict["owner_id"] = owner_id
            if document_types:
                filter_dict["document_type"] = {"$in": document_types}
            
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5, "filter": filter_dict if filter_dict else None}
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a legal AI assistant. Use the following legal documents context to provide accurate and helpful responses. 
                Always cite relevant sources and indicate when information comes from the provided context versus general legal knowledge.
                
                Context:
                {context}"""),
                ("human", "{question}")
            ])
            
            def format_docs(docs):
                return "\n\n".join(f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}" for doc in docs)
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke(question)
            
            source_docs = retriever.get_relevant_documents(question)
            sources = [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "document_id": doc.metadata.get("document_id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in source_docs
            ]
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error answering question with LangChain RAG: {str(e)}")
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
            
            logger.info(f"Updated index for document {document_id} using LangChain")
            return updated_doc_id
            
        except Exception as e:
            logger.error(f"Error updating document index {document_id} with LangChain: {str(e)}")
            raise

    async def delete_document_index(self, document_id: str):
        try:
            if document_id not in self.metadata_store:
                logger.warning(f"Document {document_id} not found in metadata store")
                return
            
            doc_metadata = self.metadata_store[document_id]
            total_chunks = doc_metadata.get("total_chunks", 0)
            
            ids_to_delete = []
            for chunk_index in range(total_chunks):
                source_id = f"doc_{document_id}_chunk_{chunk_index}"
                for doc_id, doc in self.vector_store.docstore._dict.items():
                    if doc.metadata.get("source") == source_id:
                        ids_to_delete.append(doc_id)
            
            if ids_to_delete:
                self.vector_store.delete(ids_to_delete)
            
            self._delete_metadata(document_id)
            self._save_vector_store()
            
            logger.info(f"Deleted document {document_id} from LangChain vector store")
            
        except Exception as e:
            logger.error(f"Error deleting document index {document_id} with LangChain: {str(e)}")
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            total_chunks = self.vector_store.index.ntotal if self.vector_store else 0
            
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
                "embedding_model": self.config.embedding_model.model_name,
                "vector_store_type": "LangChain_FAISS"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats with LangChain: {str(e)}")
            raise

    async def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        try:
            if not self.vector_store:
                return []
            
            all_docs = []
            for doc_id, doc in self.vector_store.docstore._dict.items():
                matches = True
                for key, value in metadata_filter.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        matches = False
                        break
                
                if matches:
                    all_docs.append({
                        "chunk_id": doc.metadata.get("source", ""),
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                    
                    if len(all_docs) >= max_results:
                        break
            
            return all_docs[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching by metadata with LangChain: {str(e)}")
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
            logger.error(f"Error getting document {document_id} with LangChain: {str(e)}")
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
            logger.error(f"Error listing documents for user {owner_id} with LangChain: {str(e)}")
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
            logger.error(f"Error getting content for document {document_id} with LangChain: {str(e)}")
            raise

    async def check_document_exists(self, content_hash: str, owner_id: int) -> Optional[str]:
        try:
            for doc_id, metadata in self.metadata_store.items():
                if (metadata.get("content_hash") == content_hash and 
                    metadata.get("owner_id") == owner_id):
                    return doc_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking document existence with LangChain: {str(e)}")
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
                "embedding_model": self.config.embedding_model.model_name,
                "vector_store_type": "LangChain_FAISS"
            }
            
        except Exception as e:
            logger.error(f"Error getting document statistics with LangChain: {str(e)}")
            raise 