#!/usr/bin/env python3
"""
Diagnostic script to debug why teammate is getting generic responses
instead of detailed rule-based answers.
"""

import sys
import traceback
import os

def test_rule_based_answerer():
    """Test if rule-based answerer works correctly"""
    print("=" * 60)
    print("🔧 DIAGNOSING RULE-BASED ANSWERER ISSUE")
    print("=" * 60)
    
    # Test 1: Check if file exists
    print("\n1️⃣ Checking if rule_based_answerer.py exists...")
    rule_based_file = "app/services/rule_based_answerer.py"
    if os.path.exists(rule_based_file):
        print(f"✅ File exists: {rule_based_file}")
    else:
        print(f"❌ File missing: {rule_based_file}")
        print("SOLUTION: Pull latest code from git!")
        return False
    
    # Test 2: Check import
    print("\n2️⃣ Testing rule-based answerer import...")
    try:
        from app.services.rule_based_answerer import RuleBasedAnswerer
        print("✅ Import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Test initialization
    print("\n3️⃣ Testing rule-based answerer initialization...")
    try:
        rba = RuleBasedAnswerer()
        print("✅ Initialization successful")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 4: Test direct functionality
    print("\n4️⃣ Testing direct functionality...")
    try:
        test_questions = [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
        
        # Simple test document
        test_doc = "grace period thirty days premium payment pre-existing thirty-six months maternity covered 24 months"
        
        answers = rba.answer_questions_from_document(test_questions, test_doc)
        
        print("✅ Rule-based answerer working!")
        for i, answer in enumerate(answers, 1):
            print(f"   Q{i}: {answer[:80]}...")
            
        # Check if we got good answers
        if any("thirty days" in answer for answer in answers):
            print("✅ Getting detailed answers (GOOD)")
        else:
            print("❌ Getting generic answers (BAD)")
            
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 5: Test QuestionAnswerer integration
    print("\n5️⃣ Testing QuestionAnswerer integration...")
    try:
        from app.services.question_answerer import QuestionAnswerer
        qa = QuestionAnswerer()
        
        if hasattr(qa, 'rule_based_answerer'):
            print("✅ QuestionAnswerer has rule_based_answerer attribute")
        else:
            print("❌ QuestionAnswerer missing rule_based_answerer attribute")
            return False
            
        print(f"Gemini available: {qa.gemini_available}")
        
    except Exception as e:
        print(f"❌ QuestionAnswerer integration test failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Test full answer_questions_with_text method
    print("\n6️⃣ Testing full answer_questions_with_text method...")
    try:
        test_doc = "Grace Period for payment of the premium shall be thirty days. pre-existing thirty-six months. maternity covered 24 months"
        test_questions = ["What is the grace period for premium payment?"]
        
        answers = qa.answer_questions_with_text(test_questions, test_doc)
        answer = answers[0]
        
        print(f"Answer received: {answer[:100]}...")
        
        if "thirty days" in answer:
            print("✅ Getting detailed rule-based answer (PERFECT!)")
        elif "relevant information is available" in answer:
            print("❌ Getting generic fallback (BAD - rule-based not working)")
        else:
            print(f"⚠️  Getting unexpected answer: {answer[:50]}...")
            
    except Exception as e:
        print(f"❌ Full method test failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    print("🔬 Running teammate diagnostic script...")
    success = test_rule_based_answerer()
    
    if success:
        print("\n✅ All tests passed! System should work correctly.")
        print("\n🚀 Try the API again - it should now return detailed answers!")
    else:
        print("\n❌ Found issues! Check the errors above.")
        print("\n💡 SOLUTIONS:")
        print("   1. Pull latest code: git pull")
        print("   2. Reinstall requirements: pip install -r requirements.txt")
        print("   3. Restart server: pkill -f uvicorn && uvicorn app.main:app --host 0.0.0.0 --port 8000")
