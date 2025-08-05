import re
import google.generativeai as genai
from typing import Dict, Any, Optional
from app.models import EntityExtraction
from app.config import settings
import logging
import json

logger = logging.getLogger(__name__)

class QueryParser:
    """Extracts structured information from natural language queries"""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    def extract_entities(self, query: str) -> EntityExtraction:
        """Extract key entities from the query using LLM and regex"""
        try:
            # First, try rule-based extraction
            rule_based_entities = self._rule_based_extraction(query)
            
            # Then, use LLM for more sophisticated extraction
            llm_entities = self._llm_based_extraction(query)
            
            # Merge results, prioritizing LLM results where available
            merged_entities = {**rule_based_entities, **llm_entities}
            
            return EntityExtraction(**merged_entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return EntityExtraction()
    
    def _rule_based_extraction(self, query: str) -> Dict[str, Any]:
        """Extract entities using regex patterns"""
        entities = {}
        query_lower = query.lower()
        
        # Age extraction
        age_pattern = r'(\d{1,3})[-\s]?(year|yr|y)?s?[-\s]?old|(\d{1,3})[mf]|\b(\d{1,3})\s*(?:year|yr|y)\b'
        age_match = re.search(age_pattern, query_lower)
        if age_match:
            age_value = next((g for g in age_match.groups() if g and g.isdigit()), None)
            if age_value:
                entities['age'] = int(age_value)
        
        # Gender extraction
        if re.search(r'\bmale\b|[0-9]+m\b', query_lower):
            entities['gender'] = 'male'
        elif re.search(r'\bfemale\b|[0-9]+f\b', query_lower):
            entities['gender'] = 'female'
        
        # Location extraction (Indian cities)
        indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata',
            'pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
            'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri', 'patna',
            'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad',
            'meerut', 'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar',
            'aurangabad', 'dhanbad', 'amritsar', 'navi mumbai', 'allahabad',
            'ranchi', 'howrah', 'coimbatore', 'jabalpur', 'gwalior', 'vijayawada'
        ]
        
        for city in indian_cities:
            if city in query_lower:
                entities['location'] = city.title()
                break
        
        # Policy duration extraction
        duration_pattern = r'(\d+)[-\s]?(month|mon|year|yr)s?[-\s]?(old\s+)?policy'
        duration_match = re.search(duration_pattern, query_lower)
        if duration_match:
            number = duration_match.group(1)
            unit = duration_match.group(2)
            entities['policy_duration'] = f"{number} {unit}{'s' if int(number) > 1 else ''}"
        
        # Medical procedures extraction
        medical_procedures = [
            'surgery', 'operation', 'procedure', 'treatment', 'therapy',
            'knee surgery', 'heart surgery', 'brain surgery', 'eye surgery',
            'dental', 'orthopedic', 'cardiac', 'neurological', 'oncology',
            'chemotherapy', 'radiation', 'dialysis', 'transplant'
        ]
        
        for procedure in medical_procedures:
            if procedure in query_lower:
                entities['procedure'] = procedure
                break
        
        # Amount extraction
        amount_pattern = r'(?:rs\.?|inr|₹)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)|(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|inr|₹|rupees?)'
        amount_match = re.search(amount_pattern, query_lower)
        if amount_match:
            amount_str = amount_match.group(1) or amount_match.group(2)
            amount_value = float(amount_str.replace(',', ''))
            entities['amount'] = amount_value
        
        return entities
    
    def _llm_based_extraction(self, query: str) -> Dict[str, Any]:
        """Extract entities using Gemini LLM"""
        try:
            prompt = f"""
            Extract structured information from this insurance/medical query. Return only a JSON object with the following fields (use null for missing information):
            {{
                "age": number or null,
                "gender": "male" or "female" or null,
                "procedure": string or null,
                "location": string or null,
                "policy_duration": string or null,
                "policy_type": string or null,
                "amount": number or null,
                "date": string or null
            }}
            
            Query: "{query}"
            
            Examples:
            - "46-year-old male, knee surgery in Pune, 3-month-old policy" → {{"age": 46, "gender": "male", "procedure": "knee surgery", "location": "Pune", "policy_duration": "3 months"}}
            - "Female patient, 35, cardiac procedure, Mumbai, new policy" → {{"age": 35, "gender": "female", "procedure": "cardiac procedure", "location": "Mumbai", "policy_duration": "new"}}
            
            Return only valid JSON:
            """
            
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                
                # Clean up the data
                cleaned_data = {}
                for key, value in parsed_data.items():
                    if value is not None and value != "null" and value != "":
                        cleaned_data[key] = value
                
                return cleaned_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in LLM-based extraction: {str(e)}")
            return {}
    
    def enhance_query(self, query: str, extracted_entities: EntityExtraction) -> str:
        """Enhance the original query with extracted entities for better search"""
        enhanced_parts = [query]
        
        if extracted_entities.age:
            enhanced_parts.append(f"age {extracted_entities.age}")
        
        if extracted_entities.gender:
            enhanced_parts.append(extracted_entities.gender)
        
        if extracted_entities.procedure:
            enhanced_parts.append(extracted_entities.procedure)
        
        if extracted_entities.location:
            enhanced_parts.append(f"location {extracted_entities.location}")
        
        if extracted_entities.policy_duration:
            enhanced_parts.append(f"policy {extracted_entities.policy_duration}")
        
        return " ".join(enhanced_parts)
    
    def generate_search_queries(self, original_query: str, entities: EntityExtraction) -> list[str]:
        """Generate multiple search queries for comprehensive retrieval"""
        queries = [original_query]
        
        # Entity-focused queries
        if entities.procedure:
            queries.append(f"{entities.procedure} coverage eligibility")
            queries.append(f"{entities.procedure} insurance policy")
        
        if entities.age and entities.procedure:
            queries.append(f"age {entities.age} {entities.procedure} coverage")
        
        if entities.location:
            queries.append(f"{entities.location} medical coverage")
            queries.append(f"network hospitals {entities.location}")
        
        if entities.policy_duration:
            queries.append(f"waiting period {entities.policy_duration}")
            queries.append(f"policy coverage {entities.policy_duration}")
        
        # General insurance terms
        queries.extend([
            "coverage exclusions limitations",
            "eligibility criteria requirements",
            "waiting period pre-existing conditions",
            "claim process approval"
        ])
        
        return list(set(queries))  # Remove duplicates