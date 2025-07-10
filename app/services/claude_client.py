from typing import List, Dict, Any, Optional
import json
import logging
from ..config import settings
from .langchain_config import (
    langchain_config,
    create_legal_document_chain,
    create_legal_analysis_chain,
    create_entity_extraction_chain,
    LegalDocumentPrompts
)

logger = logging.getLogger(__name__)

class LangChainClaudeClient:
    def __init__(self):
        self.config = langchain_config
        self.llm = self.config.anthropic_llm
        
    async def generate_legal_document(
        self,
        document_type: str,
        prompt: str,
        context_chunks: Optional[List[str]] = None,
        template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        try:
            result = await create_legal_document_chain(
                config=self.config,
                document_type=document_type,
                requirements=prompt,
                context_chunks=context_chunks,
                template=template,
                variables=variables
            )
            
            logger.info(f"Generated {document_type} document using LangChain")
            return result
            
        except Exception as e:
            logger.error(f"Error generating legal document with LangChain: {str(e)}")
            raise

    async def analyze_legal_document(
        self,
        content: str,
        analysis_type: str,
        context_chunks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        try:
            result = await create_legal_analysis_chain(
                config=self.config,
                analysis_type=analysis_type,
                content=content,
                context_chunks=context_chunks
            )
            
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {"analysis": result}
            
            logger.info(f"Completed {analysis_type} analysis using LangChain")
            return result
                
        except Exception as e:
            logger.error(f"Error analyzing legal document with LangChain: {str(e)}")
            raise

    async def extract_legal_entities(self, content: str) -> Dict[str, List[str]]:
        try:
            result = await create_entity_extraction_chain(
                config=self.config,
                content=content
            )
            
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {"entities": result}
            
            logger.info("Extracted legal entities using LangChain")
            return result
                
        except Exception as e:
            logger.error(f"Error extracting legal entities with LangChain: {str(e)}")
            raise

    async def generate_legal_summary(
        self, 
        content: str, 
        focus_areas: Optional[List[str]] = None
    ) -> str:
        try:
            focus_text = ""
            if focus_areas:
                focus_text = f"Focus specifically on these areas: {', '.join(focus_areas)}"
            
            analysis_prompt = f"""
            Provide a comprehensive legal summary of the following document. {focus_text}
            
            Include:
            1. Main purpose and type of document
            2. Key parties involved
            3. Important terms and conditions
            4. Critical dates and deadlines
            5. Rights and obligations
            6. Legal implications and considerations
            
            Document content:
            {content}
            """
            
            result = await self.llm.ainvoke([
                ("human", analysis_prompt)
            ])
            
            logger.info("Generated legal summary using LangChain")
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating legal summary with LangChain: {str(e)}")
            raise

    async def check_document_compliance(
        self,
        content: str,
        regulations: List[str],
        jurisdiction: str = "federal"
    ) -> Dict[str, Any]:
        try:
            compliance_prompt = f"""
            Review the following legal document for compliance with these regulations: {', '.join(regulations)}
            Jurisdiction: {jurisdiction}
            
            Provide a detailed compliance analysis including:
            1. Compliance status for each regulation
            2. Areas of non-compliance or concern
            3. Recommendations for improvement
            4. Risk assessment
            
            Return the results as a JSON object with this structure:
            {{
                "overall_compliance_score": 0.0-1.0,
                "regulation_compliance": {{
                    "regulation_name": {{
                        "compliant": true/false,
                        "issues": ["list of issues"],
                        "recommendations": ["list of recommendations"]
                    }}
                }},
                "risk_level": "low/medium/high",
                "summary": "overall compliance summary"
            }}
            
            Document content:
            {content}
            """
            
            result = await self.llm.ainvoke([
                ("human", compliance_prompt)
            ])
            
            try:
                return json.loads(result.content)
            except json.JSONDecodeError:
                return {"compliance_analysis": result.content}
                
        except Exception as e:
            logger.error(f"Error checking document compliance with LangChain: {str(e)}")
            raise

    async def generate_contract_amendments(
        self,
        original_contract: str,
        amendment_requests: List[str],
        context_chunks: Optional[List[str]] = None
    ) -> str:
        try:
            context_text = ""
            if context_chunks:
                context_text = f"Relevant legal context:\n{chr(10).join(context_chunks)}"
            
            amendment_prompt = f"""
            Generate contract amendments for the following requests:
            {chr(10).join([f"- {req}" for req in amendment_requests])}
            
            Original contract:
            {original_contract}
            
            {context_text}
            
            Create professional contract amendment language that:
            1. Clearly identifies what is being changed
            2. Uses proper legal formatting
            3. Maintains consistency with the original contract
            4. Includes necessary legal clauses for amendments
            5. Preserves the integrity of unchanged provisions
            """
            
            result = await self.llm.ainvoke([
                ("human", amendment_prompt)
            ])
            
            logger.info("Generated contract amendments using LangChain")
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating contract amendments with LangChain: {str(e)}")
            raise

    async def analyze_contract_risks(
        self,
        contract_content: str,
        business_context: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            context_text = ""
            if business_context:
                context_text = f"Business context: {business_context}"
            
            risk_prompt = f"""
            Perform a comprehensive risk analysis of this contract. {context_text}
            
            Analyze and identify:
            1. Financial risks and liabilities
            2. Legal compliance risks
            3. Operational risks
            4. Termination and dispute resolution risks
            5. Intellectual property risks
            6. Performance and delivery risks
            
            Return results as JSON:
            {{
                "overall_risk_score": 0.0-1.0,
                "risk_categories": {{
                    "financial": {{"score": 0.0-1.0, "issues": [], "recommendations": []}},
                    "legal": {{"score": 0.0-1.0, "issues": [], "recommendations": []}},
                    "operational": {{"score": 0.0-1.0, "issues": [], "recommendations": []}},
                    "termination": {{"score": 0.0-1.0, "issues": [], "recommendations": []}},
                    "intellectual_property": {{"score": 0.0-1.0, "issues": [], "recommendations": []}},
                    "performance": {{"score": 0.0-1.0, "issues": [], "recommendations": []}}
                }},
                "critical_issues": [],
                "mitigation_strategies": []
            }}
            
            Contract content:
            {contract_content}
            """
            
            result = await self.llm.ainvoke([
                ("human", risk_prompt)
            ])
            
            try:
                return json.loads(result.content)
            except json.JSONDecodeError:
                return {"risk_analysis": result.content}
                
        except Exception as e:
            logger.error(f"Error analyzing contract risks with LangChain: {str(e)}")
            raise

    async def generate_legal_research_queries(
        self,
        topic: str,
        jurisdiction: str = "federal",
        document_types: Optional[List[str]] = None
    ) -> List[str]:
        try:
            doc_types_text = ""
            if document_types:
                doc_types_text = f"Focus on these document types: {', '.join(document_types)}"
            
            research_prompt = f"""
            Generate effective legal research queries for the topic: {topic}
            Jurisdiction: {jurisdiction}
            {doc_types_text}
            
            Create 5-7 specific, targeted research queries that would help find:
            1. Relevant case law
            2. Applicable statutes and regulations
            3. Legal precedents
            4. Recent legal developments
            5. Practical guidance and commentary
            
            Return as a JSON array of query strings.
            """
            
            result = await self.llm.ainvoke([
                ("human", research_prompt)
            ])
            
            try:
                queries = json.loads(result.content)
                if isinstance(queries, list):
                    return queries
                else:
                    return [str(queries)]
            except json.JSONDecodeError:
                return [result.content]
                
        except Exception as e:
            logger.error(f"Error generating research queries with LangChain: {str(e)}")
            raise

    async def draft_legal_memo(
        self,
        issue: str,
        facts: str,
        legal_authorities: Optional[List[str]] = None,
        conclusion: Optional[str] = None
    ) -> str:
        try:
            authorities_text = ""
            if legal_authorities:
                authorities_text = f"Legal authorities to consider:\n{chr(10).join(legal_authorities)}"
            
            conclusion_text = ""
            if conclusion:
                conclusion_text = f"Preferred conclusion/recommendation: {conclusion}"
            
            memo_prompt = f"""
            Draft a professional legal memorandum addressing the following:
            
            Issue: {issue}
            
            Facts: {facts}
            
            {authorities_text}
            
            {conclusion_text}
            
            Structure the memorandum with:
            1. MEMORANDUM header with TO/FROM/DATE/RE
            2. ISSUE PRESENTED
            3. BRIEF ANSWER
            4. FACTS
            5. DISCUSSION (with legal analysis)
            6. CONCLUSION
            
            Use proper legal citation format and professional legal writing style.
            """
            
            result = await self.llm.ainvoke([
                ("human", memo_prompt)
            ])
            
            logger.info("Generated legal memorandum using LangChain")
            return result.content
            
        except Exception as e:
            logger.error(f"Error drafting legal memo with LangChain: {str(e)}")
            raise 