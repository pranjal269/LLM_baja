import google.generativeai as genai
from typing import List, Dict, Any
from app.models import DecisionResponse, SearchResult, EntityExtraction, ClauseReference
from app.config import settings
import logging
import json
import re

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generates structured answers using retrieved context and LLM"""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    def generate_decision(self, query: str, search_results: List[SearchResult], entities: EntityExtraction) -> DecisionResponse:
        """Generate a structured decision based on query and retrieved context"""
        try:
            # Prepare context from search results
            context = self._prepare_context(search_results)
            
            # Generate the decision using LLM
            decision_data = self._generate_llm_decision(query, context, entities)
            
            # Create clause references
            clause_references = self._create_clause_references(search_results, decision_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(search_results, decision_data)
            
            return DecisionResponse(
                decision=decision_data.get("decision", "needs_review"),
                amount=decision_data.get("amount"),
                justification=decision_data.get("justification", "Unable to determine based on available information"),
                referenced_clauses=clause_references,
                extracted_entities=entities,
                confidence_score=confidence_score,
                processing_time_ms=0  # Will be set by the caller
            )
            
        except Exception as e:
            logger.error(f"Error generating decision: {str(e)}")
            return self._create_fallback_response(entities)
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context string from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results[:5]):  # Use top 5 results
            chunk = result.chunk
            context_parts.append(
                f"[Document: {chunk.document_name}] "
                f"[Page: {chunk.page_number or 'N/A'}] "
                f"[Similarity: {result.similarity_score:.3f}]\n"
                f"{chunk.chunk_text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_llm_decision(self, query: str, context: str, entities: EntityExtraction) -> Dict[str, Any]:
        """Use LLM to generate decision based on context"""
        try:
            prompt = f"""
You are an insurance policy analyzer. Based on the provided policy documents and user query, make a decision about coverage.

QUERY: {query}

EXTRACTED ENTITIES:
- Age: {entities.age}
- Gender: {entities.gender}
- Procedure: {entities.procedure}
- Location: {entities.location}
- Policy Duration: {entities.policy_duration}
- Amount: {entities.amount}

POLICY CONTEXT:
{context}

INSTRUCTIONS:
1. Analyze the policy context to determine if the requested procedure/claim is covered
2. Consider waiting periods, exclusions, age limits, and other policy terms
3. If coverage is approved, determine the payout amount based on policy terms
4. Provide clear justification referencing specific policy clauses

Return ONLY a JSON object with this exact structure:
{{
    "decision": "approved" | "rejected" | "needs_review",
    "amount": number or null,
    "justification": "detailed explanation with specific policy references",
    "key_factors": ["list", "of", "key", "decision", "factors"],
    "referenced_sections": ["list", "of", "document", "sections", "used"]
}}

EXAMPLES:
- If procedure is covered and no exclusions apply: {{"decision": "approved", "amount": 50000, "justification": "Knee surgery is covered under Section 3.2..."}}
- If waiting period not met: {{"decision": "rejected", "amount": null, "justification": "3-month policy does not meet 6-month waiting period..."}}
- If insufficient information: {{"decision": "needs_review", "amount": null, "justification": "Additional medical documentation required..."}}

JSON Response:
"""
            
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            return {"decision": "needs_review", "justification": "Unable to parse LLM response"}
            
        except Exception as e:
            logger.error(f"Error in LLM decision generation: {str(e)}")
            return {"decision": "needs_review", "justification": f"Error processing request: {str(e)}"}
    
    def _create_clause_references(self, search_results: List[SearchResult], decision_data: Dict[str, Any]) -> List[ClauseReference]:
        """Create clause references from search results"""
        references = []
        
        try:
            for i, result in enumerate(search_results[:3]):  # Top 3 most relevant
                chunk = result.chunk
                
                # Generate a clause ID
                clause_id = f"clause_{chunk.document_name}_{chunk.chunk_index}"
                
                reference = ClauseReference(
                    clause_id=clause_id,
                    clause_text=chunk.chunk_text[:500] + "..." if len(chunk.chunk_text) > 500 else chunk.chunk_text,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    confidence_score=result.similarity_score
                )
                references.append(reference)
            
        except Exception as e:
            logger.error(f"Error creating clause references: {str(e)}")
        
        return references
    
    def _calculate_confidence(self, search_results: List[SearchResult], decision_data: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the decision"""
        try:
            # Base confidence from top search result
            base_confidence = search_results[0].similarity_score if search_results else 0.5
            
            # Adjust based on decision type
            decision = decision_data.get("decision", "needs_review")
            if decision == "approved":
                confidence_multiplier = 0.9
            elif decision == "rejected":
                confidence_multiplier = 0.85
            else:  # needs_review
                confidence_multiplier = 0.6
            
            # Adjust based on number of supporting results
            num_high_confidence_results = sum(1 for r in search_results if r.similarity_score > 0.8)
            if num_high_confidence_results >= 2:
                confidence_multiplier *= 1.1
            elif num_high_confidence_results == 0:
                confidence_multiplier *= 0.8
            
            final_confidence = min(1.0, base_confidence * confidence_multiplier)
            return round(final_confidence, 3)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _create_fallback_response(self, entities: EntityExtraction) -> DecisionResponse:
        """Create a fallback response when generation fails"""
        return DecisionResponse(
            decision="needs_review",
            amount=None,
            justification="Unable to process the request due to technical issues. Please review manually.",
            referenced_clauses=[],
            extracted_entities=entities,
            confidence_score=0.1,
            processing_time_ms=0
        )
    
    def generate_explanation(self, decision_response: DecisionResponse) -> str:
        """Generate a human-readable explanation of the decision"""
        try:
            entities = decision_response.extracted_entities
            
            explanation_parts = []
            
            # Add entity summary
            if entities.age or entities.gender or entities.procedure:
                entity_summary = "Based on the query about "
                entity_parts = []
                
                if entities.age and entities.gender:
                    entity_parts.append(f"{entities.age}-year-old {entities.gender}")
                elif entities.age:
                    entity_parts.append(f"{entities.age}-year-old patient")
                elif entities.gender:
                    entity_parts.append(f"{entities.gender} patient")
                
                if entities.procedure:
                    entity_parts.append(f"seeking {entities.procedure}")
                
                if entities.location:
                    entity_parts.append(f"in {entities.location}")
                
                explanation_parts.append(entity_summary + ", ".join(entity_parts) + ":")
            
            # Add decision
            decision_text = {
                "approved": "✅ APPROVED",
                "rejected": "❌ REJECTED", 
                "needs_review": "⚠️ NEEDS REVIEW"
            }.get(decision_response.decision, decision_response.decision.upper())
            
            explanation_parts.append(f"\n{decision_text}")
            
            # Add amount if applicable
            if decision_response.amount:
                explanation_parts.append(f"Amount: ₹{decision_response.amount:,.2f}")
            
            # Add justification
            explanation_parts.append(f"\nReason: {decision_response.justification}")
            
            # Add confidence
            explanation_parts.append(f"\nConfidence: {decision_response.confidence_score:.1%}")
            
            # Add referenced documents
            if decision_response.referenced_clauses:
                docs = set(clause.document_name for clause in decision_response.referenced_clauses)
                explanation_parts.append(f"\nReferenced Documents: {', '.join(docs)}")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Unable to generate explanation."