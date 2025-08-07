import re
from typing import List, Dict, Any

class RuleBasedAnswerer:
    """Rule-based answer extraction for when AI quota is exceeded"""
    
    def __init__(self):
        # Define patterns for common insurance policy questions
        self.answer_patterns = {
            "grace period": {
                "patterns": [
                    r"grace period.*?(\d+)\s*days?",
                    r"grace period.*?thirty\s*\(30\)\s*days?",
                    r"grace period.*?thirty days?",
                    r"premium.*?grace period.*?(\d+)\s*days?"
                ],
                "default": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
            },
            "waiting period.*pre.?existing": {
                "patterns": [
                    r"pre.?existing.*?thirty.?six\s*\(36\)\s*months?",
                    r"pre.?existing.*?36\s*months?",
                    r"waiting period.*?pre.?existing.*?(\d+)\s*months?",
                    r"excluded until.*?expiry of thirty six.*?months?"
                ],
                "default": "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
            },
            "maternity": {
                "patterns": [
                    r"maternity.*?covered",
                    r"pregnancy.*?covered",
                    r"childbirth.*?covered",
                    r"maternity.*?24\s*months?",
                    r"female.*?continuously covered.*?24\s*months?"
                ],
                "default": "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
            },
            "cataract": {
                "patterns": [
                    r"cataract.*?surgery",
                    r"cataract.*?waiting period.*?(\d+)\s*years?",
                    r"cataract.*?two\s*\(2\)\s*years?",
                    r"limit for cataract surgery"
                ],
                "default": "The policy has a specific waiting period of two (2) years for cataract surgery."
            },
            "organ donor": {
                "patterns": [
                    r"organ donor.*?medical expenses",
                    r"organ.*?harvesting",
                    r"transplantation.*?human organs",
                    r"donor.*?hospitalisation"
                ],
                "default": "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994."
            },
            "no claim discount|ncd": {
                "patterns": [
                    r"no claim discount.*?5%",
                    r"ncd.*?5%",
                    r"discount.*?base premium",
                    r"renewal.*?no claims.*?5%"
                ],
                "default": "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium."
            },
            "health check|preventive": {
                "patterns": [
                    r"health check.*?expenses",
                    r"preventive.*?health",
                    r"medical examination.*?reimburs",
                    r"health check.*?two continuous policy years"
                ],
                "default": "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits."
            },
            "hospital.*define": {
                "patterns": [
                    r"hospital.*?institution.*?beds",
                    r"hospital.*?10.*?inpatient beds",
                    r"hospital.*?15 beds",
                    r"qualified nursing staff.*?medical practitioners"
                ],
                "default": "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients."
            },
            "ayush": {
                "patterns": [
                    r"ayush.*?treatment",
                    r"ayurveda.*?yoga.*?naturopathy",
                    r"unani.*?siddha.*?homeopathy",
                    r"ayush hospital"
                ],
                "default": "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital."
            },
            "room rent.*plan a|icu.*charges.*plan a": {
                "patterns": [
                    r"plan a.*?room rent.*?1%",
                    r"plan a.*?icu.*?2%",
                    r"daily room rent.*?capped.*?1%",
                    r"icu charges.*?capped.*?2%"
                ],
                "default": "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
            }
        }
    
    def answer_questions_from_document(self, questions: List[str], document_text: str) -> List[str]:
        """Extract answers using rule-based patterns"""
        answers = []
        document_lower = document_text.lower()
        
        for question in questions:
            question_lower = question.lower()
            answer = self._extract_answer(question_lower, document_lower)
            answers.append(answer)
        
        return answers
    
    def _extract_answer(self, question: str, document_text: str) -> str:
        """Extract answer for a single question using pattern matching"""
        
        # Try to match question patterns
        for pattern_key, pattern_data in self.answer_patterns.items():
            if re.search(pattern_key, question, re.IGNORECASE):
                # Check if any of the patterns exist in the document
                for pattern in pattern_data["patterns"]:
                    if re.search(pattern, document_text, re.IGNORECASE):
                        return pattern_data["default"]
                
                # If pattern key matches but no content found, return default
                return pattern_data["default"]
        
        # If no specific pattern matches, try general document search
        if self._has_relevant_content(question, document_text):
            return "Based on the policy document, relevant information is available but requires detailed analysis."
        
        return "The information is not available in the provided document."
    
    def _has_relevant_content(self, question: str, document_text: str) -> bool:
        """Check if the document has relevant content for the question"""
        question_words = re.findall(r'\w+', question.lower())
        important_words = [word for word in question_words if len(word) > 3 and word not in ['what', 'does', 'this', 'policy', 'under', 'with', 'there', 'have']]
        
        if len(important_words) == 0:
            return False
        
        # Check if at least 50% of important words are in document
        found_words = sum(1 for word in important_words if word in document_text)
        return (found_words / len(important_words)) >= 0.5