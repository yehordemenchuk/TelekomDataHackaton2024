import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Type
import httpx
from pydantic import BaseModel, Field

from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.callbacks.manager import CallbackManagerForToolRun

from ..config import settings
from .langchain_config import langchain_config, LegalTool

logger = logging.getLogger(__name__)

class LegalDatabaseSearchInput(BaseModel):
    query: str = Field(description="Search query for legal database")
    database: str = Field(description="Database type: cases, statutes, or regulations")
    jurisdiction: str = Field(default="federal", description="Legal jurisdiction")
    date_range: Optional[str] = Field(default=None, description="Date range for search")

class LangChainLegalDatabaseTool(LegalTool):
    name: str = "legal_database_search"
    description: str = "Search legal databases for case law, statutes, and regulations"
    args_schema: Type[BaseModel] = LegalDatabaseSearchInput
    
    def _run(
        self, 
        query: str, 
        database: str, 
        jurisdiction: str = "federal",
        date_range: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            results = self._search_legal_database(query, database, jurisdiction, date_range)
            
            response = {
                "success": True,
                "results": results,
                "query": query,
                "database": database,
                "jurisdiction": jurisdiction
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in legal database search: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(
        self, 
        query: str, 
        database: str, 
        jurisdiction: str = "federal",
        date_range: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return self._run(query, database, jurisdiction, date_range, run_manager)
    
    def _search_legal_database(
        self, 
        query: str, 
        database: str, 
        jurisdiction: str, 
        date_range: Optional[str]
    ) -> List[Dict[str, Any]]:
        mock_results = [
            {
                "title": f"Case Law Result for '{query}'",
                "citation": "Example v. Case, 123 F.3d 456 (Fed. Cir. 2023)",
                "summary": f"Relevant case law regarding {query} in {jurisdiction} jurisdiction",
                "url": "https://example-legal-db.com/case/123",
                "relevance_score": 0.85,
                "database": database
            },
            {
                "title": f"Statute Related to '{query}'",
                "citation": "15 U.S.C. § 123",
                "summary": f"Federal statute addressing {query}",
                "url": "https://example-legal-db.com/statute/456",
                "relevance_score": 0.78,
                "database": database
            }
        ]
        
        return mock_results

class DocumentValidationInput(BaseModel):
    document_content: str = Field(description="Document content to validate")
    document_type: str = Field(description="Type of legal document")
    jurisdiction: str = Field(default="federal", description="Legal jurisdiction")
    validation_level: str = Field(default="basic", description="Validation level: basic, detailed, or comprehensive")

class LangChainDocumentValidationTool(LegalTool):
    name: str = "document_validation"
    description: str = "Validate legal documents for completeness and compliance"
    args_schema: Type[BaseModel] = DocumentValidationInput
    
    def _run(
        self, 
        document_content: str, 
        document_type: str, 
        jurisdiction: str = "federal",
        validation_level: str = "basic",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            validation_results = self._validate_document(
                document_content, document_type, jurisdiction, validation_level
            )
            
            response = {
                "success": True,
                "validation_results": validation_results,
                "document_type": document_type,
                "jurisdiction": jurisdiction
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in document validation: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(
        self, 
        document_content: str, 
        document_type: str, 
        jurisdiction: str = "federal",
        validation_level: str = "basic",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return self._run(document_content, document_type, jurisdiction, validation_level, run_manager)
    
    def _validate_document(
        self, 
        content: str, 
        doc_type: str, 
        jurisdiction: str, 
        validation_level: str
    ) -> Dict[str, Any]:
        validation_results = {
            "overall_score": 0.85,
            "issues": [],
            "recommendations": [],
            "required_sections": [],
            "missing_elements": [],
            "compliance_level": validation_level
        }
        
        type_requirements = {
            "contract": ["parties", "consideration", "terms", "signatures"],
            "will": ["testator", "executor", "beneficiaries", "signature", "witnesses"],
            "lease": ["landlord", "tenant", "property", "rent", "term"],
            "nda": ["parties", "confidential information", "obligations", "term"],
            "power_of_attorney": ["principal", "attorney", "powers", "limitations"]
        }
        
        if doc_type.lower() in type_requirements:
            required_sections = type_requirements[doc_type.lower()]
            validation_results["required_sections"] = required_sections
            
            missing = []
            for section in required_sections:
                if section.lower() not in content.lower():
                    missing.append(section)
            
            validation_results["missing_elements"] = missing
            
            if missing:
                validation_results["issues"].append(f"Missing required sections: {', '.join(missing)}")
                validation_results["recommendations"].append("Add all required document sections")
                validation_results["overall_score"] -= 0.1 * len(missing)
        
        validation_results["overall_score"] = max(0.0, min(1.0, validation_results["overall_score"]))
        
        return validation_results

class CitationCheckInput(BaseModel):
    text: str = Field(description="Text containing citations to check")
    citation_format: str = Field(default="bluebook", description="Citation format: bluebook, apa, mla")
    auto_correct: bool = Field(default=False, description="Whether to auto-correct citations")

class LangChainCitationCheckerTool(LegalTool):
    name: str = "citation_checker"
    description: str = "Check and format legal citations according to standard citation formats"
    args_schema: Type[BaseModel] = CitationCheckInput
    
    def _run(
        self, 
        text: str, 
        citation_format: str = "bluebook",
        auto_correct: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            citation_results = self._check_citations(text, citation_format, auto_correct)
            
            response = {
                "success": True,
                "citation_analysis": citation_results,
                "format": citation_format,
                "auto_corrected": auto_correct
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in citation checking: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(
        self, 
        text: str, 
        citation_format: str = "bluebook",
        auto_correct: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return self._run(text, citation_format, auto_correct, run_manager)
    
    def _check_citations(
        self, 
        text: str, 
        citation_format: str, 
        auto_correct: bool
    ) -> Dict[str, Any]:
        import re
        
        citation_patterns = {
            "case_law": r'\b\w+\s+v\.\s+\w+.*?\d+.*?\(\d{4}\)',
            "statute": r'\b\d+\s+U\.S\.C\.\s+§\s+\d+',
            "regulation": r'\b\d+\s+C\.F\.R\.\s+§\s+\d+',
            "court_rule": r'\bFed\.\s+R\.\s+\w+\.\s+P\.\s+\d+'
        }
        
        found_citations = []
        citation_issues = []
        
        for citation_type, pattern in citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                found_citations.append({
                    "text": match,
                    "type": citation_type,
                    "format_compliant": True
                })
        
        corrected_text = text
        if auto_correct:
            for citation in found_citations:
                if not citation["format_compliant"]:
                    corrected_citation = self._format_citation(citation["text"], citation_format)
                    corrected_text = corrected_text.replace(citation["text"], corrected_citation)
        
        return {
            "citations_found": len(found_citations),
            "citations": found_citations,
            "issues": citation_issues,
            "corrected_text": corrected_text if auto_correct else None,
            "format_compliance_score": len([c for c in found_citations if c["format_compliant"]]) / max(1, len(found_citations))
        }
    
    def _format_citation(self, citation_text: str, format_style: str) -> str:
        if format_style.lower() == "bluebook":
            return citation_text
        elif format_style.lower() == "apa":
            return citation_text
        else:
            return citation_text

class ContractAnalysisInput(BaseModel):
    contract_content: str = Field(description="Contract content to analyze")
    analysis_type: str = Field(description="Type of analysis: risk, terms, compliance")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific areas to focus on")

class LangChainContractAnalysisTool(LegalTool):
    name: str = "contract_analysis"
    description: str = "Analyze contracts for risks, terms, and compliance issues"
    args_schema: Type[BaseModel] = ContractAnalysisInput
    
    def _run(
        self, 
        contract_content: str, 
        analysis_type: str,
        focus_areas: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            analysis_results = self._analyze_contract(contract_content, analysis_type, focus_areas)
            
            response = {
                "success": True,
                "analysis_results": analysis_results,
                "analysis_type": analysis_type,
                "focus_areas": focus_areas
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"Error in contract analysis: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(
        self, 
        contract_content: str, 
        analysis_type: str,
        focus_areas: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return self._run(contract_content, analysis_type, focus_areas, run_manager)
    
    def _analyze_contract(
        self, 
        content: str, 
        analysis_type: str, 
        focus_areas: Optional[List[str]]
    ) -> Dict[str, Any]:
        if analysis_type == "risk":
            return self._analyze_contract_risks(content, focus_areas)
        elif analysis_type == "terms":
            return self._analyze_contract_terms(content, focus_areas)
        elif analysis_type == "compliance":
            return self._analyze_contract_compliance(content, focus_areas)
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
    
    def _analyze_contract_risks(self, content: str, focus_areas: Optional[List[str]]) -> Dict[str, Any]:
        risks = {
            "financial_risks": [],
            "legal_risks": [],
            "operational_risks": [],
            "overall_risk_score": 0.3
        }
        
        risk_keywords = {
            "financial": ["penalty", "liquidated damages", "indemnity", "liability"],
            "legal": ["governing law", "jurisdiction", "dispute resolution", "termination"],
            "operational": ["performance", "delivery", "service level", "milestones"]
        }
        
        for risk_type, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    risks[f"{risk_type}_risks"].append(f"Contains {keyword} provisions")
        
        return risks
    
    def _analyze_contract_terms(self, content: str, focus_areas: Optional[List[str]]) -> Dict[str, Any]:
        terms = {
            "payment_terms": [],
            "performance_terms": [],
            "termination_terms": [],
            "key_obligations": []
        }
        
        term_patterns = {
            "payment": ["payment", "invoice", "fee", "cost", "price"],
            "performance": ["deliver", "perform", "complete", "service"],
            "termination": ["terminate", "end", "expire", "cancel"]
        }
        
        for term_type, keywords in term_patterns.items():
            found_terms = []
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    found_terms.append(keyword)
            terms[f"{term_type}_terms"] = found_terms
        
        return terms
    
    def _analyze_contract_compliance(self, content: str, focus_areas: Optional[List[str]]) -> Dict[str, Any]:
        compliance = {
            "regulatory_compliance": [],
            "standard_clauses": [],
            "missing_provisions": [],
            "compliance_score": 0.8
        }
        
        standard_clauses = ["force majeure", "confidentiality", "governing law", "signatures"]
        
        for clause in standard_clauses:
            if clause.lower() in content.lower():
                compliance["standard_clauses"].append(clause)
            else:
                compliance["missing_provisions"].append(clause)
        
        compliance["compliance_score"] = len(compliance["standard_clauses"]) / len(standard_clauses)
        
        return compliance

class LangChainMCPService:
    def __init__(self):
        self.config = langchain_config
        self.llm = self.config.anthropic_llm
        self.tools = []
        self._register_default_tools()
    
    def _register_default_tools(self):
        self.tools = [
            LangChainLegalDatabaseTool(),
            LangChainDocumentValidationTool(),
            LangChainCitationCheckerTool(),
            LangChainContractAnalysisTool()
        ]
        logger.info(f"Registered {len(self.tools)} LangChain tools")
    
    def register_tool(self, tool: BaseTool):
        self.tools.append(tool)
        logger.info(f"Registered custom tool: {tool.name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.schema() if hasattr(tool, 'args_schema') else {}
            }
            for tool in self.tools
        ]
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tool = self._get_tool_by_name(tool_name)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found"
                }
            
            if hasattr(tool, '_arun'):
                result = await tool._arun(**parameters)
            else:
                result = tool._run(**parameters)
            
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {"success": True, "result": result}
            
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def create_legal_agent(self) -> AgentExecutor:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a legal AI assistant with access to specialized legal tools. 
                You can search legal databases, validate documents, check citations, and analyze contracts.
                
                Always use the appropriate tools to provide accurate and comprehensive legal assistance.
                When using tools, explain what you're doing and interpret the results for the user.
                
                Available tools:
                - legal_database_search: Search for case law, statutes, and regulations
                - document_validation: Validate legal documents for completeness
                - citation_checker: Check and format legal citations
                - contract_analysis: Analyze contracts for risks and terms
                
                Remember to cite sources and provide disclaimers about seeking professional legal advice."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=3,
                return_intermediate_steps=True
            )
            
            logger.info("Created LangChain legal agent with tools")
            return agent_executor
            
        except Exception as e:
            logger.error(f"Error creating legal agent: {str(e)}")
            raise
    
    async def run_agent_query(self, query: str) -> Dict[str, Any]:
        try:
            agent_executor = await self.create_legal_agent()
            
            result = await agent_executor.ainvoke({
                "input": query
            })
            
            return {
                "success": True,
                "response": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error running agent query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def batch_execute_tools(
        self, 
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            parameters = tool_call.get("parameters", {})
            
            result = await self.execute_tool(tool_name, parameters)
            result["tool_name"] = tool_name
            results.append(result)
        
        return results
    
    async def suggest_tools_for_task(self, task_description: str) -> List[str]:
        task_lower = task_description.lower()
        suggested_tools = []
        
        if any(keyword in task_lower for keyword in ["search", "find", "case", "statute", "law"]):
            suggested_tools.append("legal_database_search")
        
        if any(keyword in task_lower for keyword in ["validate", "check", "review", "compliance"]):
            suggested_tools.append("document_validation")
        
        if any(keyword in task_lower for keyword in ["citation", "cite", "format", "reference"]):
            suggested_tools.append("citation_checker")
        
        if any(keyword in task_lower for keyword in ["contract", "agreement", "analyze", "risk"]):
            suggested_tools.append("contract_analysis")
        
        return suggested_tools
    
    async def get_tool_help(self, tool_name: str) -> Dict[str, Any]:
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}
        
        help_info = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema.schema() if hasattr(tool, 'args_schema') else {},
            "examples": self._get_tool_examples(tool_name)
        }
        
        return help_info
    
    def _get_tool_examples(self, tool_name: str) -> List[Dict[str, Any]]:
        examples = {
            "legal_database_search": [
                {
                    "description": "Search for contract law cases",
                    "parameters": {
                        "query": "breach of contract",
                        "database": "cases",
                        "jurisdiction": "federal"
                    }
                }
            ],
            "document_validation": [
                {
                    "description": "Validate a contract document",
                    "parameters": {
                        "document_content": "This agreement is between...",
                        "document_type": "contract",
                        "validation_level": "detailed"
                    }
                }
            ],
            "citation_checker": [
                {
                    "description": "Check citations in legal text",
                    "parameters": {
                        "text": "As stated in Brown v. Board of Education...",
                        "citation_format": "bluebook",
                        "auto_correct": True
                    }
                }
            ],
            "contract_analysis": [
                {
                    "description": "Analyze contract for risks",
                    "parameters": {
                        "contract_content": "Service agreement between...",
                        "analysis_type": "risk",
                        "focus_areas": ["liability", "termination"]
                    }
                }
            ]
        }
        
        return examples.get(tool_name, []) 