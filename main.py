#!/usr/bin/env python3
"""
🛡️ ALGOSPEAK CONTENT MODERATION API

Complete API using our simple normalizer + classifier:
- normalizer.py: algospeak → normal text
- classifier.py: text → harmful/safe classification

Usage: POST /moderate with {"text": "I want to unalive myself"}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

# Import our components (now in root directory)
from normalizer import SimpleNormalizer
try:
    from classifier import SimpleClassifier
    classifier_available = True
except ImportError as e:
    print(f"⚠️ Classifier not available: {e}")
    classifier_available = False

app = FastAPI(
    title="🛡️ Algospeak Content Moderation API",
    description="""
    ## Simple Two-Stage Content Moderation
    
    **Stage 1**: Normalize algospeak (unalive → kill)
    **Stage 2**: AI classification (kill myself → extremely_harmful)
    
    **Clean & Simple Architecture!**
    """,
    version="1.0.0"
)

class ModerationRequest(BaseModel):
    text: str

class ModerationResponse(BaseModel):
    original_text: str
    normalized_text: str
    algospeak_detected: bool
    classification: str
    stage1_status: str
    stage2_status: str

# Initialize components
print("🔧 Initializing Algospeak Moderation API...")
normalizer = SimpleNormalizer()
if classifier_available:
    classifier = SimpleClassifier()
    print("✅ Both normalizer and classifier loaded!")
else:
    print("⚠️ Only normalizer loaded, classifier needs Ollama")

@app.get("/")
async def root():
    """API health check."""
    return {
        "message": "🛡️ Algospeak Content Moderation API",
        "status": "active",
        "architecture": {
            "normalizer": "normalizer.py (103 lines)",
            "classifier": "classifier.py (145 lines)", 
            "api": "main.py (this file)"
        },
        "workflow": {
            "1": "POST /moderate with text",
            "2": "Normalize algospeak patterns", 
            "3": "AI classification via Ollama",
            "4": "Return complete results"
        },
        "example": {
            "input": "I want to unalive myself",
            "normalized": "I want to kill myself",
            "classification": "extremely_harmful, self_harm"
        }
    }

@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(request: ModerationRequest):
    """
    Complete moderation pipeline: Algospeak detection + AI classification.
    """
    try:
        # Stage 1: Normalize algospeak
        print(f"📝 Input: {request.text}")
        normalized_text = normalizer.normalize(request.text)
        algospeak_detected = normalized_text != request.text
        
        if algospeak_detected:
            stage1_status = "algospeak_normalized"
            print(f"🔄 Normalized algospeak detected")
        else:
            stage1_status = "no_algospeak_found"
            print(f"➡️ No algospeak patterns found")
        
        # Stage 2: AI Classification (if available)
        if classifier_available:
            print(f"🤖 Running AI classification...")
            classification_result = classifier.classify(request.text)
            
            if 'error' in classification_result:
                stage2_status = "ollama_unavailable"
                classification = f"⚠️ {classification_result['error']}"
                print(f"❌ Classification failed: {classification_result['error']}")
            else:
                stage2_status = "ai_classified"
                classification = str(classification_result.get('classification', 'Unknown'))
                print(f"✅ AI classification completed")
        else:
            stage2_status = "classifier_not_loaded"
            classification = "Classifier module not available"
            print(f"⚠️ Classifier not available")
        
        result = ModerationResponse(
            original_text=request.text,
            normalized_text=normalized_text,
            algospeak_detected=algospeak_detected,
            classification=classification,
            stage1_status=stage1_status,
            stage2_status=stage2_status
        )
        
        print(f"�� Complete: {stage1_status} + {stage2_status}")
        return result
        
    except Exception as e:
        print(f"❌ Error in moderation pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")

@app.get("/demo")
async def demo():
    """Demo endpoint showing normalizer working."""
    test_cases = [
        "I want to unalive myself",
        "This is seggs content", 
        "I killed it at work today",
        "Great job on that presentation!"
    ]
    
    results = []
    for text in test_cases:
        try:
            normalized = normalizer.normalize(text)
            results.append({
                "input": text,
                "normalized": normalized,
                "algospeak_detected": normalized != text,
                "status": "✅ Working"
            })
        except Exception as e:
            results.append({
                "input": text,
                "error": str(e),
                "status": "❌ Error"
            })
    
    return {
        "demo": "Algospeak Moderation Pipeline Demo",
        "test_results": results,
        "note": "AI classification requires trained model via Ollama",
        "files": {
            "normalizer": "normalizer.py",
            "classifier": "classifier.py", 
            "patterns": "finetunning/dataset/algospeak_patterns.json"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    normalizer_working = False
    classifier_working = False
    
    try:
        # Test normalizer
        test_result = normalizer.normalize("unalive")
        normalizer_working = test_result == "kill"
    except Exception as e:
        pass
    
    try:
        # Test classifier (if available)
        if classifier_available:
            # Just check if it initializes
            classifier_working = True
    except Exception as e:
        pass
        
    return {
        "api": "healthy",
        "normalizer": "✅ working" if normalizer_working else "❌ error",
        "classifier": "✅ loaded" if classifier_working else "⚠️ unavailable", 
        "patterns_loaded": len(getattr(normalizer, 'patterns', {})),
        "project_structure": {
            "normalizer.py": "103 lines",
            "classifier.py": "145 lines",
            "main.py": "~180 lines",
            "total": "~428 lines (vs 2700+ before!)"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Algospeak Content Moderation API...")
    print("📋 Available endpoints:")
    print("   GET  /       - API info & examples")
    print("   POST /moderate - Moderate content")
    print("   GET  /demo   - Test normalizer")
    print("   GET  /health - System health check")
    print("   GET  /docs  - API documentation")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)



