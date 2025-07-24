"""
🛡️ STAGE 2: STEP 4 - PRODUCTION INFERENCE MODULE

📋 EXECUTION FLOW (ML Training Pipeline):
1. 📊 1_dataset_prep.py prepares training data first
2. 🤖 2_training.py fine-tunes LLM on prepared data
3. 📈 3_evaluation.py validates model performance
4. ⚡ THIS FILE (4_inference.py) provides production classification
5. 🚀 5_stage2_demo.py demonstrates the complete AI system

PURPOSE: Production-ready inference using trained LLM for real-time content classification.
CALLED BY: main.py (API) for live content moderation
USES: Models trained by 2_training.py and validated by 3_evaluation.py
INTEGRATES: Stage 1 normalization for preprocessing
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from ..algospeak import AlgospeakNormalizer, NormalizationResult


@dataclass
class ModerationResult:
    """Result of content moderation inference."""
    
    # Input
    original_text: str
    normalized_text: str
    normalization_applied: bool
    
    # Classification results
    label: str
    confidence: float
    category: str
    severity: str
    
    # Detailed scores
    label_probabilities: Dict[str, float]
    category_probabilities: Dict[str, float]
    
    # Algospeak detection
    algospeak_matches: List[Dict[str, Any]]
    
    # Action recommendation
    recommended_action: str
    reasoning: str
    
    # Metadata
    model_version: str
    processing_time_ms: float


class ContentModerationInference:
    """Inference engine for content moderation."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        confidence_threshold: float = 0.7,
        severe_threshold: float = 0.9
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.severe_threshold = severe_threshold
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize normalizer
        self.normalizer = AlgospeakNormalizer()
        
        # Label mappings
        self.id_to_label = {
            0: "safe",
            1: "potentially_harmful", 
            2: "harmful",
            3: "extremely_harmful"
        }
        
        self.id_to_category = {
            0: "none",
            1: "hate_speech",
            2: "self_harm",
            3: "adult_content",
            4: "violence",
            5: "harassment",
            6: "misinformation",
            7: "spam"
        }
        
        self.severity_mapping = {
            "safe": "none",
            "potentially_harmful": "low",
            "harmful": "high", 
            "extremely_harmful": "extreme"
        }
        
        print(f"✅ Content moderation model loaded from {model_path}")
        print(f"🔧 Using device: {self.device}")
    
    def predict_single(self, text: str) -> ModerationResult:
        """Predict moderation label for a single text."""
        import time
        start_time = time.time()
        
        # Normalize algospeak
        normalization_result = self.normalizer.normalize_text(text)
        input_text = normalization_result.normalized_text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=-1)
            label_probs = probabilities[0].cpu().numpy()
            
            # Get predictions
            predicted_label_id = torch.argmax(logits, dim=-1).item()
            predicted_label = self.id_to_label[predicted_label_id]
            confidence = float(label_probs[predicted_label_id])
        
        # Create probability distributions
        label_probabilities = {
            label: float(prob) for label, prob in 
            zip(self.id_to_label.values(), label_probs)
        }
        
        # Determine category (simplified - could be separate model)
        category = self._determine_category(text, normalization_result.matches)
        category_probabilities = {category: 1.0}  # Simplified
        
        # Determine severity
        severity = self.severity_mapping[predicted_label]
        
        # Recommend action
        action, reasoning = self._recommend_action(
            predicted_label, confidence, severity, normalization_result.matches
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return ModerationResult(
            original_text=text,
            normalized_text=input_text,
            normalization_applied=len(normalization_result.matches) > 0,
            label=predicted_label,
            confidence=confidence,
            category=category,
            severity=severity,
            label_probabilities=label_probabilities,
            category_probabilities=category_probabilities,
            algospeak_matches=[match.__dict__ for match in normalization_result.matches],
            recommended_action=action,
            reasoning=reasoning,
            model_version="1.0.0",
            processing_time_ms=processing_time
        )
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[ModerationResult]:
        """Predict moderation labels for a batch of texts."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.predict_single(text) for text in batch]
            results.extend(batch_results)
        
        return results
    
    def _determine_category(self, text: str, algospeak_matches: List) -> str:
        """Determine content category based on content and algospeak matches."""
        # Simplified category detection based on keywords and algospeak
        text_lower = text.lower()
        
        # Check algospeak matches first
        for match in algospeak_matches:
            target = match.replacement.lower()
            if any(word in target for word in ["kill", "die", "suicide", "harm"]):
                return "self_harm"
            elif any(word in target for word in ["sex", "porn", "adult"]):
                return "adult_content"
            elif any(word in target for word in ["hate", "slur"]):
                return "hate_speech"
            elif any(word in target for word in ["violence", "gun", "weapon"]):
                return "violence"
        
        # Check original text
        if any(word in text_lower for word in ["kill", "die", "suicide", "harm", "hurt"]):
            return "self_harm"
        elif any(word in text_lower for word in ["hate", "stupid", "idiot"]):
            return "harassment"
        elif any(word in text_lower for word in ["buy", "sale", "discount", "offer"]):
            return "spam"
        
        return "none"
    
    def _recommend_action(
        self, 
        label: str, 
        confidence: float, 
        severity: str,
        algospeak_matches: List
    ) -> Tuple[str, str]:
        """Recommend moderation action based on prediction."""
        
        # High confidence extreme content
        if label == "extremely_harmful" and confidence > self.severe_threshold:
            return "auto_block", f"Extremely harmful content detected with high confidence ({confidence:.2f})"
        
        # High confidence harmful content
        if label == "harmful" and confidence > self.confidence_threshold:
            if len(algospeak_matches) > 0:
                return "auto_block", f"Harmful content with algospeak evasion detected ({confidence:.2f})"
            else:
                return "flag_for_review", f"Harmful content detected ({confidence:.2f})"
        
        # Potentially harmful content
        if label == "potentially_harmful":
            if confidence > self.confidence_threshold:
                return "flag_for_review", f"Potentially harmful content ({confidence:.2f})"
            else:
                return "monitor", f"Low confidence potentially harmful content ({confidence:.2f})"
        
        # Safe content
        if label == "safe":
            if len(algospeak_matches) > 0:
                return "monitor", "Safe content but contains algospeak patterns"
            else:
                return "allow", f"Safe content ({confidence:.2f})"
        
        # Default fallback
        return "flag_for_review", f"Uncertain classification ({confidence:.2f})"
    
    def explain_prediction(self, result: ModerationResult) -> Dict[str, Any]:
        """Provide detailed explanation of the prediction."""
        explanation = {
            "input_analysis": {
                "original_text": result.original_text,
                "normalized_text": result.normalized_text,
                "normalization_applied": result.normalization_applied,
                "algospeak_patterns_found": len(result.algospeak_matches)
            },
            
            "model_prediction": {
                "predicted_label": result.label,
                "confidence": result.confidence,
                "all_label_probabilities": result.label_probabilities
            },
            
            "algospeak_detection": {
                "patterns_detected": [
                    {
                        "original": match["original_text"],
                        "replacement": match["replacement"],
                        "confidence": match["confidence"],
                        "severity": match["severity_level"]
                    }
                    for match in result.algospeak_matches
                ]
            },
            
            "decision_logic": {
                "recommended_action": result.recommended_action,
                "reasoning": result.reasoning,
                "category": result.category,
                "severity": result.severity
            },
            
            "performance": {
                "processing_time_ms": result.processing_time_ms,
                "model_version": result.model_version
            }
        }
        
        return explanation


def load_model_for_inference(
    model_path: str,
    confidence_threshold: float = 0.7,
    severe_threshold: float = 0.9
) -> ContentModerationInference:
    """Load trained model for inference."""
    return ContentModerationInference(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        severe_threshold=severe_threshold
    )


if __name__ == "__main__":
    # Example usage
    try:
        # Load model (this will fail until we have a trained model)
        inference = load_model_for_inference("models/content_moderation_with_normalization")
        
        # Test examples
        test_texts = [
            "I want to unalive myself",
            "This game is killer!",
            "I hate this weather",
            "Buy my product now with 50% off!",
            "That seggs scene was inappropriate"
        ]
        
        for text in test_texts:
            result = inference.predict_single(text)
            print(f"\nText: {text}")
            print(f"Label: {result.label} (confidence: {result.confidence:.3f})")
            print(f"Action: {result.recommended_action}")
            print(f"Reasoning: {result.reasoning}")
            
    except Exception as e:
        print(f"Model not found (expected until training is complete): {e}")
        print("Run training.py first to create the model!") 