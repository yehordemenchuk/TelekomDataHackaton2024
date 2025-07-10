import anthropic
from typing import List, Dict, Any, Optional
import json
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        
    async def generate_legal_document(
        self,
        document_type: str,
        prompt: str,
        context_chunks: Optional[List[str]] = None,
        template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a legal document using Claude."""
        try:
            system_prompt = self._build_legal_system_prompt(document_type)
            user_prompt = self._build_generation_prompt(
                document_type, prompt, context_chunks, template, variables
            )
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating legal document: {str(e)}")
            raise

    async def analyze_legal_document(
        self,
        content: str,
        analysis_type: str,
        context_chunks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze a legal document for various aspects."""
        try:
            system_prompt = self._build_analysis_system_prompt(analysis_type)
            user_prompt = self._build_analysis_prompt(content, analysis_type, context_chunks)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for analysis
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Try to parse as JSON, fallback to text
            try:
                return json.loads(response.content[0].text)
            except json.JSONDecodeError:
                return {"analysis": response.content[0].text}
                
        except Exception as e:
            logger.error(f"Error analyzing legal document: {str(e)}")
            raise

    async def extract_legal_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract legal entities from document content."""
        try:
            system_prompt = """You are a legal document analysis expert. Extract key legal entities from the provided document content."""
            
            user_prompt = f"""
            Extract the following legal entities from this document:
            - Parties (individuals, companies, organizations)
            - Dates (important dates, deadlines, effective dates)
            - Financial amounts (payments, penalties, fees)
            - Legal terms and clauses
            - Jurisdictions and governing law
            - Contract terms and conditions
            
            Return the results as a JSON object with each entity type as a key and a list of extracted entities as values.
            
            Document content:
            {content}
            """
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            try:
                return json.loads(response.content[0].text)
            except json.JSONDecodeError:
                return {"entities": response.content[0].text}
                
        except Exception as e:
            logger.error(f"Error extracting legal entities: {str(e)}")
            raise

    def _build_legal_system_prompt(self, document_type: str) -> str:
        """Build system prompt for legal document generation."""
        base_prompt = """You are an expert legal document assistant specializing in creating high-quality, professional legal documents. 
        You have deep knowledge of legal terminology, document structure, and best practices for various types of legal documents.
        
        Always ensure that:
        1. Documents are well-structured and professionally formatted
        2. Legal language is precise and appropriate
        3. All necessary clauses and sections are included
        4. Documents comply with general legal standards
        5. Placeholder text is clearly marked for user customization
        
        Important: These documents should be reviewed by qualified legal professionals before use."""
        
        document_specific = {
            "contract": "Focus on creating comprehensive contracts with clear terms, conditions, and obligations.",
            "agreement": "Create formal agreements with proper legal structure and mutual obligations.",
            "declaration": "Draft clear, factual declarations with appropriate legal language.",
            "motion": "Prepare formal legal motions with proper citations and legal arguments.",
            "brief": "Create well-researched legal briefs with strong arguments and citations.",
            "memorandum": "Draft professional legal memoranda with clear analysis and recommendations.",
            "will": "Prepare comprehensive wills with proper legal formalities and clarity.",
            "power_of_attorney": "Create detailed power of attorney documents with specific powers and limitations.",
            "lease": "Draft comprehensive lease agreements with clear terms and conditions.",
            "nda": "Prepare thorough non-disclosure agreements with appropriate scope and protections."
        }
        
        specific_guidance = document_specific.get(document_type, "Create a professional legal document.")
        return f"{base_prompt}\n\n{specific_guidance}"

    def _build_generation_prompt(
        self,
        document_type: str,
        prompt: str,
        context_chunks: Optional[List[str]] = None,
        template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt for document generation."""
        user_prompt = f"Create a {document_type} based on the following requirements:\n\n{prompt}\n\n"
        
        if template:
            user_prompt += f"Use this template as a starting point:\n{template}\n\n"
        
        if variables:
            user_prompt += f"Include these specific details:\n{json.dumps(variables, indent=2)}\n\n"
        
        if context_chunks:
            user_prompt += "Use the following relevant legal precedents and context:\n"
            for i, chunk in enumerate(context_chunks, 1):
                user_prompt += f"\nContext {i}:\n{chunk}\n"
        
        user_prompt += "\nPlease generate a complete, professional legal document."
        return user_prompt

    def _build_analysis_system_prompt(self, analysis_type: str) -> str:
        """Build system prompt for legal document analysis."""
        base_prompt = """You are an expert legal analyst with deep knowledge of legal documents, contracts, and legal principles. 
        Provide thorough, accurate analysis with attention to legal nuances and potential issues."""
        
        analysis_specific = {
            "summary": "Provide clear, concise summaries that capture key points and legal implications.",
            "risk_assessment": "Identify potential legal risks, liabilities, and areas of concern with recommendations.",
            "compliance_check": "Review for compliance with relevant laws, regulations, and legal standards.",
            "clause_extraction": "Extract and categorize key clauses, terms, and provisions with explanations."
        }
        
        specific_guidance = analysis_specific.get(analysis_type, "Provide thorough legal analysis.")
        return f"{base_prompt}\n\n{specific_guidance}"

    def _build_analysis_prompt(
        self,
        content: str,
        analysis_type: str,
        context_chunks: Optional[List[str]] = None
    ) -> str:
        """Build user prompt for document analysis."""
        prompts = {
            "summary": "Provide a comprehensive summary of this legal document, highlighting key terms, obligations, and important provisions.",
            "risk_assessment": "Analyze this document for potential legal risks, liabilities, and areas of concern. Provide recommendations for mitigation.",
            "compliance_check": "Review this document for compliance with relevant laws and regulations. Identify any potential compliance issues.",
            "clause_extraction": "Extract and categorize all important clauses, terms, and provisions from this document. Explain their legal significance."
        }
        
        user_prompt = f"{prompts.get(analysis_type, 'Analyze this legal document.')}\n\n"
        user_prompt += f"Document content:\n{content}\n\n"
        
        if context_chunks:
            user_prompt += "Consider this additional legal context:\n"
            for i, chunk in enumerate(context_chunks, 1):
                user_prompt += f"\nContext {i}:\n{chunk}\n"
        
        user_prompt += f"\nProvide a detailed {analysis_type} in JSON format with structured results."
        return user_prompt 