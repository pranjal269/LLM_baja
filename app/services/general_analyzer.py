import re
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class GeneralDocumentAnalyzer:
    """General-purpose document analyzer that works with any document type"""
    
    def __init__(self):
        # Common document type indicators
        self.document_types = {
            "insurance_policy": ["policy", "premium", "coverage", "claim", "insured", "deductible"],
            "contract": ["agreement", "party", "terms", "conditions", "obligations", "breach"],
            "manual": ["instructions", "procedure", "step", "guide", "how to", "manual"],
            "report": ["findings", "analysis", "results", "conclusion", "summary", "data"],
            "legal": ["shall", "whereas", "therefore", "pursuant", "covenant", "liability"],
            "medical": ["patient", "treatment", "diagnosis", "symptoms", "medication", "prescription"],
            "financial": ["revenue", "profit", "loss", "investment", "financial", "budget"],
            "technical": ["specification", "technical", "system", "configuration", "requirements"]
        }
        
        # Question type patterns
        self.question_types = {
            "what_is": ["what is", "what does", "what are"],
            "how_to": ["how to", "how do", "how can", "how should"],
            "when": ["when", "what time", "at what point"],
            "where": ["where", "in which", "at which location"],
            "why": ["why", "what reason", "what purpose"],
            "who": ["who", "which person", "what entity"],
            "list": ["list", "what are all", "enumerate", "name all"],
            "definition": ["define", "definition of", "meaning of", "what does mean"],
            "process": ["process", "procedure", "steps", "how does work"],
            "rules": ["rules", "regulations", "requirements", "policies about"]
        }
    
    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of any document"""
        try:
            analysis = {
                "document_type": self._identify_document_type(document_text),
                "main_topics": self._extract_main_topics(document_text),
                "key_sections": self._identify_key_sections(document_text),
                "document_summary": self._generate_summary(document_text),
                "key_entities": self._extract_key_entities(document_text),
                "document_length": len(document_text),
                "structure": self._analyze_structure(document_text)
            }
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {
                "document_type": "unknown",
                "main_topics": [],
                "key_sections": {},
                "document_summary": "Document analysis failed",
                "key_entities": [],
                "document_length": len(document_text),
                "structure": {"sections": 0, "paragraphs": 0}
            }
    
    def answer_question(self, question: str, document_text: str, document_analysis: Dict[str, Any] = None) -> str:
        """Answer any question about any document using intelligent content analysis"""
        try:
            if not document_analysis:
                document_analysis = self.analyze_document(document_text)
            
            question_lower = question.lower().strip()
            question_type = self._classify_question(question_lower)
            
            # Handle different question types
            if question_type == "what_is" and any(word in question_lower for word in ["document", "about", "this"]):
                return self._answer_document_about(document_analysis, document_text)
            
            elif question_type == "what_is" and "main topics" in question_lower:
                return self._answer_main_topics(document_analysis)
            
            elif question_type == "list" or "topics" in question_lower:
                return self._answer_main_topics(document_analysis)
            
            elif "rules" in question_lower or "regulations" in question_lower:
                return self._answer_rules_question(question, document_text, document_analysis)
            
            else:
                # General semantic search for any other question
                return self._answer_general_question(question, document_text, document_analysis)
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Unable to process the question due to technical difficulties."
    
    def _identify_document_type(self, document_text: str) -> str:
        """Identify the type of document based on content"""
        text_lower = document_text.lower()
        
        type_scores = {}
        for doc_type, keywords in self.document_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[doc_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return "general_document"
    
    def _extract_main_topics(self, document_text: str) -> List[str]:
        """Extract main topics from the document"""
        # Clean and process text
        text_lower = document_text.lower()
        
        # Remove common stop words and extract meaningful phrases
        sentences = re.split(r'[.!?]+', document_text)
        topics = []
        
        # Look for headings and important phrases
        heading_patterns = [
            r'^[A-Z][A-Z\s\-:]+$',  # All caps headings
            r'^\d+\.?\s+[A-Z][^.!?]*$',  # Numbered headings
            r'^[A-Z][^.!?]*:',  # Colon headings
        ]
        
        for sentence in sentences[:20]:  # Focus on first 20 sentences for topics
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 100:
                for pattern in heading_patterns:
                    if re.match(pattern, sentence):
                        topics.append(sentence.strip())
                        break
        
        # Extract key noun phrases using simple patterns
        key_phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', document_text)
        phrase_counts = Counter(key_phrases)
        
        # Add most common meaningful phrases
        for phrase, count in phrase_counts.most_common(10):
            if len(phrase) > 3 and count > 1:
                topics.append(phrase)
        
        return list(set(topics))[:10]  # Return top 10 unique topics
    
    def _identify_key_sections(self, document_text: str) -> Dict[str, str]:
        """Identify key sections of the document"""
        sections = {}
        
        # Split by common section markers
        parts = re.split(r'(?:\n|^)(?:\d+\.?\s*|[A-Z]+\.?\s*)?([A-Z][A-Z\s\-:]{3,})\s*(?:\n|$)', document_text)
        
        current_section = "introduction"
        current_content = []
        
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Section header
                if current_content:
                    sections[current_section] = ' '.join(current_content).strip()[:500]
                current_section = part.strip().lower()
                current_content = []
            else:  # Section content
                if part.strip():
                    current_content.append(part.strip())
        
        # Add the last section
        if current_content:
            sections[current_section] = ' '.join(current_content).strip()[:500]
        
        return sections
    
    def _generate_summary(self, document_text: str) -> str:
        """Generate a summary of the document"""
        # Take first and last paragraphs for summary
        paragraphs = [p.strip() for p in document_text.split('\n\n') if len(p.strip()) > 50]
        
        if not paragraphs:
            return "Document content is not sufficiently structured for summary generation."
        
        # Use first paragraph and key sentences
        summary_parts = []
        
        if paragraphs:
            # First paragraph often contains key information
            first_para = paragraphs[0][:300]
            summary_parts.append(first_para)
        
        # Look for sentences with key terms
        sentences = re.split(r'[.!?]+', document_text)
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 30 and len(sentence) < 200 and
                any(word in sentence.lower() for word in ['shall', 'must', 'required', 'important', 'key', 'main'])):
                key_sentences.append(sentence)
                if len(key_sentences) >= 3:
                    break
        
        if key_sentences:
            summary_parts.extend(key_sentences)
        
        return ' '.join(summary_parts)[:800]
    
    def _extract_key_entities(self, document_text: str) -> List[str]:
        """Extract key entities like names, organizations, etc."""
        entities = []
        
        # Find capitalized entities
        entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Ltd|Inc|Corp|Company|Policy))\b',  # Companies
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Person names (3 words)
            r'\b[A-Z]{2,}\b',  # Abbreviations
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, document_text)
            entities.extend(matches)
        
        # Remove duplicates and return most common
        entity_counts = Counter(entities)
        return [entity for entity, count in entity_counts.most_common(10)]
    
    def _analyze_structure(self, document_text: str) -> Dict[str, int]:
        """Analyze document structure"""
        return {
            "sections": len(re.findall(r'(?:\n|^)[A-Z][A-Z\s\-:]{3,}', document_text)),
            "paragraphs": len([p for p in document_text.split('\n\n') if len(p.strip()) > 20]),
            "sentences": len(re.split(r'[.!?]+', document_text)),
            "words": len(document_text.split())
        }
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked"""
        for q_type, patterns in self.question_types.items():
            if any(pattern in question for pattern in patterns):
                return q_type
        return "general"
    
    def _answer_document_about(self, analysis: Dict[str, Any], document_text: str) -> str:
        """Answer 'What is this document about?' questions"""
        doc_type = analysis.get("document_type", "unknown")
        summary = analysis.get("document_summary", "")
        main_topics = analysis.get("main_topics", [])
        
        response_parts = []
        
        # Document type
        type_descriptions = {
            "insurance_policy": "This is an insurance policy document",
            "contract": "This is a contractual agreement document",
            "manual": "This is an instructional manual or guide",
            "report": "This is a report or analysis document",
            "legal": "This is a legal document",
            "medical": "This is a medical document",
            "financial": "This is a financial document",
            "technical": "This is a technical specification document",
            "general_document": "This is a general document"
        }
        
        response_parts.append(type_descriptions.get(doc_type, "This is a document"))
        
        # Add summary if available
        if summary and len(summary) > 50:
            response_parts.append(f"that {summary[:200]}...")
        
        # Add main topics
        if main_topics:
            topics_text = ", ".join(main_topics[:5])
            response_parts.append(f"It covers topics such as: {topics_text}.")
        
        result = " ".join(response_parts)
        return self._clean_response_text(result)
    
    def _answer_main_topics(self, analysis: Dict[str, Any]) -> str:
        """Answer questions about main topics"""
        main_topics = analysis.get("main_topics", [])
        key_sections = analysis.get("key_sections", {})
        
        if not main_topics and not key_sections:
            return "Unable to identify specific topics from the document content."
        
        response_parts = ["The main topics covered in this document include:"]
        
        # Add topics from analysis
        if main_topics:
            for i, topic in enumerate(main_topics[:8], 1):
                response_parts.append(f"{i}. {topic}")
        
        # Add section topics if available
        if key_sections and not main_topics:
            for i, section_name in enumerate(key_sections.keys(), 1):
                if i <= 8:
                    response_parts.append(f"{i}. {section_name.title()}")
        
        result = "\n".join(response_parts)
        return self._clean_response_text(result)
    
    def _answer_rules_question(self, question: str, document_text: str, analysis: Dict[str, Any]) -> str:
        """Answer questions about rules, regulations, requirements"""
        # Search for rule-related content
        rule_patterns = [
            r'[Ss]hall\s+[^.!?]*[.!?]',
            r'[Mm]ust\s+[^.!?]*[.!?]',
            r'[Rr]equired\s+[^.!?]*[.!?]',
            r'[Pp]rohibited\s+[^.!?]*[.!?]',
            r'[Nn]ot\s+permitted\s+[^.!?]*[.!?]',
            r'[Rr]ules?\s+[^.!?]*[.!?]',
            r'[Rr]egulations?\s+[^.!?]*[.!?]'
        ]
        
        rules = []
        for pattern in rule_patterns:
            matches = re.findall(pattern, document_text)
            rules.extend(matches[:3])  # Limit to 3 per pattern
        
        # Extract question keywords to find relevant rules
        question_keywords = re.findall(r'\b[a-z]+\b', question.lower())
        question_keywords = [word for word in question_keywords if len(word) > 3]
        
        relevant_rules = []
        for rule in rules:
            rule_lower = rule.lower()
            if any(keyword in rule_lower for keyword in question_keywords):
                relevant_rules.append(rule.strip())
        
        if relevant_rules:
            result = f"Based on the document, here are the relevant rules:\n\n" + "\n\n".join(relevant_rules[:3])
            return self._clean_response_text(result)
        elif rules:
            result = f"Here are some key rules from the document:\n\n" + "\n\n".join(rules[:3])
            return self._clean_response_text(result)
        else:
            return "No specific rules or regulations were found in the document related to your query."
    
    def _answer_general_question(self, question: str, document_text: str, analysis: Dict[str, Any]) -> str:
        """Answer general questions using keyword matching and context extraction"""
        # Extract keywords from question
        question_words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        question_keywords = [word for word in question_words if len(word) > 3 and 
                           word not in ['what', 'where', 'when', 'who', 'why', 'how', 'does', 'this', 'that', 'with', 'from', 'they', 'have', 'been']]
        
        if not question_keywords:
            return "Unable to identify key terms in your question to search the document."
        
        # Find relevant sentences
        sentences = re.split(r'[.!?]+', document_text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 500:
                sentence_lower = sentence.lower()
                # Score based on keyword matches
                score = sum(1 for keyword in question_keywords if keyword in sentence_lower)
                if score > 0:
                    relevant_sentences.append((score, sentence))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sentence for score, sentence in relevant_sentences[:3]]
        
        if top_sentences:
            result = " ".join(top_sentences)
            return self._clean_response_text(result)
        else:
            # Fallback: provide document summary
            result = analysis.get("document_summary", "No specific information found in the document related to your question.")
            return self._clean_response_text(result)
    
    def _clean_response_text(self, text: str) -> str:
        """Clean up response text to remove unnecessary quotes and formatting issues"""
        try:
            import re
            
            # Remove excessive quotation marks around terms - be more aggressive
            # Replace quoted terms like "Accident" with just Accident (but preserve necessary quotes)
            text = re.sub(r'"([A-Za-z][A-Za-z\s]*?)"', r'\1', text)
            
            # Remove quotes around single words if they're clearly not needed
            text = re.sub(r'"([A-Za-z]{2,})"', r'\1', text)
            
            # Remove quotes around common terms and phrases
            text = re.sub(r'"([A-Z][a-z]+ [A-Z][a-z]+)"', r'\1', text)  # "Global Health"
            text = re.sub(r'"([A-Z]{2,}[A-Z\s]*)"', r'\1', text)  # "AYUSH"
            
            # Clean up extra spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Remove quotes from the beginning and end if they wrap the whole text
            text = text.strip()
            if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
                text = text[1:-1].strip()
            
            # Remove document artifacts and page markers
            text = re.sub(r'--- Page \d+ ---', '', text)
            text = re.sub(r'Page \d+ of \d+', '', text)
            text = re.sub(r'UIN[-:]?\s*[A-Z0-9]+', '', text)  # Remove policy numbers
            
            # Clean up multiple spaces again after artifact removal
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            # If cleaning fails, return original text
            return text
