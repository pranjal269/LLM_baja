import google.generativeai as genai
from typing import List, Dict, Any
from app.models import SearchResult, DocumentChunk
from app.config import settings
import logging
import re

logger = logging.getLogger(__name__)

class QuestionAnswerer:
    """Service to answer multiple questions based on document content"""
    
    def __init__(self):
        try:
            if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your-gemini-api-key-here":
                genai.configure(api_key=settings.GEMINI_API_KEY)
                
                # Directly use models/gemini-pro - this is the actual model name expected by the API
                try:
                    self.model = genai.GenerativeModel("models/gemini-pro")
                    self.gemini_available = True
                    logger.info("✅ Successfully initialized Gemini with models/gemini-pro")
                except Exception as e:
                    logger.error(f"Failed with models/gemini-pro: {str(e)}")
                    # Try without models/ prefix as fallback
                    try:
                        self.model = genai.GenerativeModel("gemini-pro")
                        self.gemini_available = True
                        logger.info("✅ Successfully initialized Gemini with gemini-pro")
                    except Exception as e2:
                        logger.error(f"Failed with gemini-pro: {str(e2)}")
                        self.model = None
                        self.gemini_available = False
            else:
                self.model = None
                self.gemini_available = False
                logger.warning("⚠️ Gemini API key not configured. Using fallback responses.")
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.model = None
            self.gemini_available = False

    def answer_questions(self, questions: List[str], search_results: List[SearchResult]) -> List[str]:
        """Answer multiple questions based on search results"""
        try:
            logger.info(f"📝 Answering {len(questions)} questions with {len(search_results)} search results")
            
            if not self.gemini_available:
                logger.warning("⚠️  Gemini not available, using fallback")
                return self._generate_fallback_answers(questions)
            
            # Prepare context from search results
            context = self._prepare_context(search_results)
            logger.debug(f"📄 Context prepared: {len(context)} characters")
            
            # Generate answers for each question
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"❓ Processing question {i+1}/{len(questions)}: {question[:50]}...")
                answer = self._answer_single_question(question, context)
                answers.append(answer)
                logger.info(f"✅ Answer {i+1}: {answer[:100]}...")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error answering questions: {str(e)}")
            return self._generate_fallback_answers(questions)

    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """Prepare context string from search results"""
        if not search_results:
            logger.warning("⚠️  No search results provided for context")
            return "No relevant document content found."
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
            chunk_text = result.chunk.chunk_text[:1000]  # Limit chunk size
            document_name = result.chunk.document_name
            similarity = result.similarity_score
            
            context_parts.append(
                f"[Context {i}] (Document: {document_name}, Relevance: {similarity:.2f})\n"
                f"{chunk_text}\n"
            )
        
        full_context = "\n---\n".join(context_parts)
        logger.debug(f"📄 Context contains {len(context_parts)} chunks, {len(full_context)} total characters")
        return full_context

    def _answer_single_question(self, question: str, context: str) -> str:
        """Answer a single question using LLM"""
        try:
            if not self.gemini_available:
                logger.warning("⚠️  Gemini not available for question answering")
                return self._extract_relevant_content(question, context)
            
            prompt = f"""
You are an expert document analyst. Answer the question based ONLY on the document context below.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- You MUST provide an answer based on the context provided.
- Extract and synthesize the most relevant information from the context.
- If exact information isn't explicitly stated, provide the closest relevant information.
- NEVER say the information is unavailable or not in the document.
- Be concise but informative.

ANSWER:"""

            logger.debug(f"🤖 Sending prompt to Gemini (length: {len(prompt)})")
            
            try:
                response = self.model.generate_content(prompt)
                answer = response.text.strip() if response and response.text else ""
            except Exception as api_error:
                logger.error(f"API error: {str(api_error)}")
                return self._extract_relevant_content(question, context)
            
            # Clean up the answer
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
            
            # Check for empty or error responses
            if not answer or "not available in" in answer.lower() or "unable to" in answer.lower():
                answer = self._extract_relevant_content(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer for question '{question}': {str(e)}")
            return self._extract_relevant_content(question, context)

    def _extract_relevant_content(self, question: str, context: str) -> str:
        """Return up to 3 relevant, meaningful sentences from the context based on keyword overlap with the question."""
        try:
            # If context is empty, return a clear message
            if not context or context.strip() == "":
                return "No relevant information found in the document."
            
            # Clean context
            cleaned_context = re.sub(r'---\s*Page\s+\d+\s*---', '', context)
            # Remove lines that look like contact info, addresses, or policy numbers
            cleaned_context = re.sub(r'(www\.[^\s]+|E[- ]?mail:|Call at:|Toll Free|Policy Wordings|UIN-|Issuing Office:|For more details, log on to:|Sales - \d+|Service - \d+)', '', cleaned_context, flags=re.IGNORECASE)
            
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_context)
            question_words = set(re.findall(r'\w+', question.lower()))
            
            # Score each sentence by number of overlapping keywords
            scored_sentences = []
            for sentence in sentences:
                sentence_clean = sentence.strip()
                # Filter out short, junk, or header sentences
                if len(sentence_clean) < 30 or re.match(r'^[A-Z\s\-:]+$', sentence_clean):
                    continue
                sentence_words = set(re.findall(r'\w+', sentence_clean.lower()))
                score = len(question_words & sentence_words)
                if score > 0:
                    scored_sentences.append((score, sentence_clean))
            
            # Sort by score (descending) and pick top 3
            scored_sentences.sort(reverse=True)
            top_sentences = [s for _, s in scored_sentences[:3]]
            
            if top_sentences:
                return " ".join(top_sentences)
            
            # If no good match, fallback to first 2 meaningful sentences
            fallback = []
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if len(sentence_clean) > 40 and not re.match(r'^[A-Z\s\-:]+$', sentence_clean):
                    fallback.append(sentence_clean)
                if len(fallback) == 2:
                    break
            if fallback:
                return " ".join(fallback)
            
            return "No relevant information found in the document."
        except Exception as e:
            logger.error(f"Error extracting relevant content: {str(e)}")
            return "No relevant information found in the document."

    def _generate_fallback_answers(self, questions: List[str]) -> List[str]:
        """Generate generic answers when the LLM is not available"""
        return [self._extract_relevant_content(question, "") for question in questions]
    
    def answer_questions_with_text(self, questions: List[str], document_text: str) -> List[str]:
        """Answer questions directly from document text"""
        try:
            logger.info(f"📝 Answering {len(questions)} questions with full text ({len(document_text)} chars)")
            
            if not self.gemini_available:
                return self._generate_fallback_answers(questions)
            
            answers = []
            for i, question in enumerate(questions):
                logger.info(f"❓ Processing question {i+1}/{len(questions)} with full text")
                answer = self._answer_with_full_text(question, document_text)
                answers.append(answer)
            
            return answers
            
        except Exception as e:
            logger.error(f"Error answering questions with text: {str(e)}")
            return self._generate_fallback_answers(questions)
    
    def _answer_with_full_text(self, question: str, document_text: str) -> str:
        """Answer a question using the full document text"""
        try:
            if not self.gemini_available:
                return self._extract_relevant_content(question, document_text)
            
            # Truncate document if too long
            max_context_length = 6000
            if len(document_text) > max_context_length:
                # Instead of just truncating, take beginning and end portions
                start_text = document_text[:4000]
                end_text = document_text[-2000:]
                document_text = start_text + "\n[...document continues...]\n" + end_text
                logger.info(f"📄 Truncated document to {len(document_text)} chars with beginning and end portions")
            
            prompt = f"""
You are an expert document analyst. Answer the question based ONLY on the document below.

DOCUMENT:
{document_text}

QUESTION: {question}

INSTRUCTIONS:
- You MUST provide an answer based on the document provided.
- Extract and synthesize the most relevant information from the document.
- If exact information isn't explicitly stated, provide the closest relevant information.
- NEVER say the information is unavailable or not in the document.
- Be concise but informative.

ANSWER:"""

            logger.debug(f"🤖 Sending full text prompt to Gemini")
            try:
                response = self.model.generate_content(prompt)
                answer = response.text.strip() if response and response.text else ""
            except Exception as api_error:
                logger.error(f"API error with full text: {str(api_error)}")
                return self._extract_relevant_content(question, document_text)
            
            # Clean up the answer
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
            
            # Check for empty or error responses
            if not answer or "not available in" in answer.lower() or "unable to" in answer.lower():
                # Use the fallback method with the document text as context
                return self._extract_relevant_content(question, document_text)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with full text: {str(e)}")
            return self._extract_relevant_content(question, document_text)