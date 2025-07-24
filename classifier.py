#!/usr/bin/env python3
"""
🛡️ SIMPLE STAGE 2: AI Content Classification

GOAL: Classify normalized text using our fine-tuned Qwen2.5-3B model
INPUT: "I want to kill myself" (from Stage 1)
OUTPUT: "extremely_harmful, self_harm, severity: 3"

Simple approach: Call Ollama API with fine-tuned model
"""

import requests
import json
from typing import Dict, Optional
import sys
from pathlib import Path

# Add stage1 to path for normalizer
from normalizer import SimpleNormalizer

class SimpleClassifier:
    """Simple AI classifier using Ollama with fine-tuned model."""
    
    def __init__(self, model_name: str = "qwen-algospeak"):
        """
        Initialize classifier with Ollama model.
        
        Args:
            model_name: Name of your fine-tuned model in Ollama
        """
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.normalizer = SimpleNormalizer()
        
        # Test if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama is running")
                models = response.json()
                model_names = [m['name'] for m in models.get('models', [])]
                if model_name in model_names:
                    print(f"✅ Model '{model_name}' found")
                else:
                    print(f"⚠️  Model '{model_name}' not found. Available: {model_names}")
            else:
                print(f"❌ Ollama not responding (status {response.status_code})")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
    
    def classify(self, text: str) -> Dict[str, str]:
        """
        Classify content as harmful or safe.
        
        Args:
            text: Input text (can contain algospeak)
            
        Returns:
            Dictionary with classification results
        """
        # Step 1: Normalize algospeak
        normalized_text = self.normalizer.normalize(text)
        
        # Step 2: Create instruction prompt for fine-tuned model
        prompt = f"""Analyze this content for harmfulness and classify it appropriately. Consider both direct language and coded language (algospeak).

Input: {normalized_text}

Classify as one of: safe, potentially_harmful, harmful, extremely_harmful
Provide category: general, self_harm, hate_speech, violence, sexual_content, harassment
Provide severity: 1 (low) to 3 (high)
"""

        # Step 3: Call Ollama API
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "max_tokens": 100
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                classification = result.get('response', '').strip()
                
                # Parse the response
                return {
                    'original_text': text,
                    'normalized_text': normalized_text,
                    'classification': classification,
                    'model_used': self.model_name,
                    'normalization_applied': text != normalized_text
                }
            else:
                return {
                    'original_text': text,
                    'normalized_text': normalized_text,
                    'error': f"Ollama API error: {response.status_code}",
                    'classification': 'error'
                }
                
        except Exception as e:
            return {
                'original_text': text,
                'normalized_text': normalized_text,
                'error': f"Classification failed: {str(e)}",
                'classification': 'error'
            }

def classify_text(text: str, model_name: str = "qwen-algospeak") -> Dict[str, str]:
    """Simple function to classify text."""
    classifier = SimpleClassifier(model_name)
    return classifier.classify(text)

# Quick test (only works if Ollama is running with your model)
if __name__ == "__main__":
    classifier = SimpleClassifier()
    
    test_cases = [
        "I want to unalive myself",
        "I killed it at work today", 
        "Great job on the presentation!"
    ]
    
    print("\n🧪 TESTING SIMPLE CLASSIFIER:")
    print("=" * 50)
    
    for text in test_cases:
        print(f"\n📝 Input: '{text}'")
        result = classifier.classify(text)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🔄 Normalized: '{result['normalized_text']}'")
            print(f"🎯 Classification: {result['classification']}")
