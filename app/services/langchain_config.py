from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from typing import Dict, Any, List, Optional, Type
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class LangChainConfig:
    def __init__(self):
        self.anthropic_llm = self._create_anthropic_llm()
        self.embedding_model = self._create_embedding_model()
        self.text_splitter = self._create_text_splitter()
        
    def _create_anthropic_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature
        )
    
    def _create_embedding_model(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.max_chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

class LegalDocumentPrompts:
    @staticmethod
    def get_legal_generation_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert legal document assistant specializing in creating high-quality, professional legal documents. 
            You have deep knowledge of legal terminology, document structure, and best practices for various types of legal documents.
            
            Always ensure that:
            1. Documents are well-structured and professionally formatted
            2. Legal language is precise and appropriate
            3. All necessary clauses and sections are included
            4. Documents comply with general legal standards
            5. Placeholder text is clearly marked for user customization
            
            Important: These documents should be reviewed by qualified legal professionals before use.
            
            Document Type Guidance:
            - Contracts: Focus on comprehensive terms, conditions, and obligations
            - Agreements: Create formal agreements with proper legal structure
            - Declarations: Draft clear, factual declarations with appropriate legal language
            - Motions: Prepare formal legal motions with proper citations and arguments
            - Briefs: Create well-researched legal briefs with strong arguments
            - Memoranda: Draft professional legal memoranda with clear analysis
            - Wills: Prepare comprehensive wills with proper legal formalities
            - Power of Attorney: Create detailed documents with specific powers and limitations
            - Leases: Draft comprehensive lease agreements with clear terms
            - NDAs: Prepare thorough non-disclosure agreements with appropriate scope"""),
            ("human", """Create a {document_type} based on the following requirements:

            Requirements: {requirements}

            {context_section}

            {template_section}

            {variables_section}

            Please generate a complete, professional legal document.""")
        ])
    
    @staticmethod
    def get_legal_analysis_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert legal analyst with deep knowledge of legal documents, contracts, and legal principles. 
            Provide thorough, accurate analysis with attention to legal nuances and potential issues.
            
            Analysis Type Guidelines:
            - Summary: Provide clear, concise summaries that capture key points and legal implications
            - Risk Assessment: Identify potential legal risks, liabilities, and areas of concern with recommendations
            - Compliance Check: Review for compliance with relevant laws, regulations, and legal standards
            - Clause Extraction: Extract and categorize key clauses, terms, and provisions with explanations
            
            Return your analysis in a structured JSON format when appropriate."""),
            ("human", """Perform a {analysis_type} analysis of the following legal document:

            Document Content:
            {content}

            {context_section}

            Please provide a thorough analysis following the guidelines for {analysis_type}.""")
        ])
    
    @staticmethod
    def get_entity_extraction_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a legal document analysis expert. Extract key legal entities from the provided document content.
            
            Return the results as a JSON object with the following structure:
            {
                "parties": ["list of individuals, companies, organizations"],
                "dates": ["important dates, deadlines, effective dates"],
                "financial_amounts": ["payments, penalties, fees"],
                "legal_terms": ["legal terms and clauses"],
                "jurisdictions": ["jurisdictions and governing law"],
                "contract_terms": ["contract terms and conditions"],
                "other_entities": ["any other relevant entities"]
            }"""),
            ("human", """Extract legal entities from this document:

            {content}""")
        ])

class LegalRAGChain:
    def __init__(self, config: LangChainConfig, vector_store: FAISS):
        self.config = config
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
    def create_qa_chain(self) -> RetrievalQA:
        prompt = PromptTemplate(
            template="""Use the following pieces of legal context to answer the question. 
            If you don't know the answer based on the context, just say that you don't know.

            Context:
            {context}

            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.config.anthropic_llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def create_rag_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal AI assistant. Use the following legal documents context to provide accurate and helpful responses. 
            Always cite relevant sources and indicate when information comes from the provided context versus general legal knowledge.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.config.anthropic_llm
            | StrOutputParser()
        )
        
        return rag_chain

class LegalTool(BaseTool):
    name: str = "legal_tool"
    description: str = "Base class for legal tools"
    
    def _run(self, **kwargs) -> str:
        raise NotImplementedError
        
    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError

def create_legal_document_chain(
    config: LangChainConfig,
    document_type: str,
    requirements: str,
    context_chunks: Optional[List[str]] = None,
    template: Optional[str] = None,
    variables: Optional[Dict[str, Any]] = None
):
    prompt = LegalDocumentPrompts.get_legal_generation_prompt()
    
    context_section = ""
    if context_chunks:
        context_section = "Relevant Legal Context:\n" + "\n\n".join(context_chunks)
    
    template_section = ""
    if template:
        template_section = f"Template to follow:\n{template}"
    
    variables_section = ""
    if variables:
        variables_section = f"Specific details to include:\n{variables}"
    
    chain = prompt | config.anthropic_llm | StrOutputParser()
    
    return chain.invoke({
        "document_type": document_type,
        "requirements": requirements,
        "context_section": context_section,
        "template_section": template_section,
        "variables_section": variables_section
    })

def create_legal_analysis_chain(
    config: LangChainConfig,
    analysis_type: str,
    content: str,
    context_chunks: Optional[List[str]] = None
):
    prompt = LegalDocumentPrompts.get_legal_analysis_prompt()
    
    context_section = ""
    if context_chunks:
        context_section = "Additional Legal Context:\n" + "\n\n".join(context_chunks)
    
    if analysis_type in ["risk_assessment", "compliance_check", "clause_extraction"]:
        chain = prompt | config.anthropic_llm | JsonOutputParser()
    else:
        chain = prompt | config.anthropic_llm | StrOutputParser()
    
    return chain.invoke({
        "analysis_type": analysis_type,
        "content": content,
        "context_section": context_section
    })

def create_entity_extraction_chain(config: LangChainConfig, content: str):
    prompt = LegalDocumentPrompts.get_entity_extraction_prompt()
    chain = prompt | config.anthropic_llm | JsonOutputParser()
    
    return chain.invoke({"content": content})

langchain_config = LangChainConfig() 