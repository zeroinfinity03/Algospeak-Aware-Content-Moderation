"""
🛡️ STAGE 1: STEP 1 - ALGOSPEAK NORMALIZATION MODULE

📋 EXECUTION FLOW:
1. 🚪 main.py (API) receives user text
2. ✨ THIS FILE (1_normalizer.py) is called FIRST to orchestrate text cleaning
3. 🔍 This file calls 2_detector.py to find algospeak patterns
4. 📊 Detector uses 3_patterns.py to load pattern database

PURPOSE: Orchestrates the conversion of algospeak patterns to canonical forms.
CALLED BY: main.py (API entry point)
CALLS: 2_detector.py (to find patterns)
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from detector import AlgospeakDetector, AlgospeakMatch
from patterns import CHAR_SUBSTITUTIONS

@dataclass
class NormalizationResult:
    """Results of text normalization process."""
    original_text: str          # Original input text
    normalized_text: str        # Text after normalization
    matches_found: List[AlgospeakMatch]  # All detected patterns
    transformations: List[Dict]  # Log of all transformations applied
    confidence_score: float     # Overall confidence in normalization
    
class AlgospeakNormalizer:
    """Main normalizer class for converting algospeak to canonical forms."""
    
    def __init__(self):
        """Initialize the normalizer with a detector."""
        self.detector = AlgospeakDetector()
        
    def normalize_text(self, text: str, preserve_case: bool = True) -> NormalizationResult:
        """
        Normalize algospeak patterns in the given text.
        
        Args:
            text: Input text to normalize
            preserve_case: Whether to preserve original case when possible
            
        Returns:
            NormalizationResult containing normalized text and metadata
        """
        # Detect all algospeak patterns
        matches = self.detector.detect_patterns(text)
        
        if not matches:
            return NormalizationResult(
                original_text=text,
                normalized_text=text,
                matches_found=[],
                transformations=[],
                confidence_score=1.0  # High confidence when no changes needed
            )
        
        # Apply normalizations
        normalized_text = text
        transformations = []
        
        # Sort matches by position (reverse order to maintain indices)
        sorted_matches = sorted(matches, key=lambda x: x.start_pos, reverse=True)
        
        for match in sorted_matches:
            # Apply the transformation
            before = normalized_text[match.start_pos:match.end_pos]
            after = match.normalized
            
            # Preserve case if requested
            if preserve_case:
                after = self._preserve_case(before, after)
            
            # Replace the text
            normalized_text = (normalized_text[:match.start_pos] + 
                             after + 
                             normalized_text[match.end_pos:])
            
            # Log the transformation
            transformations.append({
                'position': (match.start_pos, match.end_pos),
                'before': before,
                'after': after,
                'pattern_type': match.pattern_type,
                'confidence': match.confidence
            })
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence(matches)
        
        return NormalizationResult(
            original_text=text,
            normalized_text=normalized_text,
            matches_found=matches,
            transformations=transformations,
            confidence_score=confidence_score
        )
    
    def normalize_batch(self, texts: List[str], preserve_case: bool = True) -> List[NormalizationResult]:
        """
        Normalize a batch of texts efficiently.
        
        Args:
            texts: List of input texts to normalize
            preserve_case: Whether to preserve original case
            
        Returns:
            List of NormalizationResult objects
        """
        return [self.normalize_text(text, preserve_case) for text in texts]
    
    def quick_normalize(self, text: str) -> str:
        """
        Quick normalization that returns only the normalized text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        result = self.normalize_text(text)
        return result.normalized_text
    
    def _preserve_case(self, original: str, normalized: str) -> str:
        """
        Preserve the case pattern of the original text in the normalized version.
        
        Args:
            original: Original text with case pattern
            normalized: Normalized text to apply case pattern to
            
        Returns:
            Normalized text with preserved case pattern
        """
        if len(original) != len(normalized):
            # For different lengths, use simple heuristics
            if original.isupper():
                return normalized.upper()
            elif original.islower():
                return normalized.lower()
            elif original.istitle():
                return normalized.title()
            else:
                return normalized
        
        # For same length, preserve character-by-character case
        result = []
        for orig_char, norm_char in zip(original, normalized):
            if orig_char.isupper():
                result.append(norm_char.upper())
            else:
                result.append(norm_char.lower())
        
        return ''.join(result)
    
    def _calculate_confidence(self, matches: List[AlgospeakMatch]) -> float:
        """
        Calculate overall confidence in the normalization process.
        
        Args:
            matches: List of detected matches
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not matches:
            return 1.0
        
        # Weight confidence by match quality
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for match in matches:
            # Higher confidence patterns get more weight
            weight = match.confidence
            total_weighted_confidence += match.confidence * weight
            total_weight += weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def get_normalization_stats(self, results: List[NormalizationResult]) -> Dict[str, any]:
        """
        Generate statistics about normalization results.
        
        Args:
            results: List of normalization results
            
        Returns:
            Dictionary containing various statistics
        """
        if not results:
            return {}
        
        total_texts = len(results)
        texts_with_algospeak = sum(1 for r in results if r.matches_found)
        total_matches = sum(len(r.matches_found) for r in results)
        
        # Pattern type distribution
        pattern_types = {}
        severity_distribution = {}
        
        for result in results:
            for match in result.matches_found:
                pattern_types[match.pattern_type] = pattern_types.get(match.pattern_type, 0) + 1
                severity_distribution[match.severity] = severity_distribution.get(match.severity, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r.confidence_score for r in results) / total_texts
        
        return {
            'total_texts_processed': total_texts,
            'texts_with_algospeak': texts_with_algospeak,
            'algospeak_detection_rate': texts_with_algospeak / total_texts * 100,
            'total_patterns_found': total_matches,
            'avg_patterns_per_text': total_matches / total_texts,
            'pattern_type_distribution': pattern_types,
            'severity_distribution': severity_distribution,
            'average_confidence': avg_confidence,
            'high_confidence_normalizations': sum(1 for r in results if r.confidence_score > 0.8)
        }
    
    def compare_normalization_impact(self, texts: List[str]) -> Dict[str, any]:
        """
        Compare text characteristics before and after normalization.
        
        Args:
            texts: List of input texts
            
        Returns:
            Comparison statistics
        """
        results = self.normalize_batch(texts)
        
        original_texts = [r.original_text for r in results]
        normalized_texts = [r.normalized_text for r in results]
        
        # Character-level changes
        total_chars_changed = 0
        texts_modified = 0
        
        for orig, norm in zip(original_texts, normalized_texts):
            if orig != norm:
                texts_modified += 1
                # Count character differences
                total_chars_changed += sum(1 for a, b in zip(orig, norm) if a != b)
                total_chars_changed += abs(len(orig) - len(norm))  # Length differences
        
        # Word-level changes
        original_words = set(word.lower() for text in original_texts for word in text.split())
        normalized_words = set(word.lower() for text in normalized_texts for word in text.split())
        
        words_added = normalized_words - original_words
        words_removed = original_words - normalized_words
        
        return {
            'texts_modified': texts_modified,
            'modification_rate': texts_modified / len(texts) * 100,
            'total_characters_changed': total_chars_changed,
            'avg_chars_changed_per_text': total_chars_changed / len(texts),
            'unique_words_added': len(words_added),
            'unique_words_removed': len(words_removed),
            'word_vocabulary_change': (len(words_added) - len(words_removed)),
            'sample_words_added': list(words_added)[:10],
            'sample_words_removed': list(words_removed)[:10]
        }

# Utility functions for advanced normalization
def apply_character_substitution(text: str) -> str:
    """
    Apply character-level substitutions (like leetspeak normalization).
    
    Args:
        text: Input text
        
    Returns:
        Text with character substitutions applied
    """
    result = text
    for char, replacement in CHAR_SUBSTITUTIONS.items():
        result = result.replace(char, replacement)
    return result

def clean_repeated_characters(text: str, max_repeats: int = 2) -> str:
    """
    Clean up excessive character repetition (e.g., "sooooo" -> "soo").
    
    Args:
        text: Input text
        max_repeats: Maximum number of repeated characters to allow
        
    Returns:
        Text with excessive repetition cleaned up
    """
    # Pattern to match 3+ repeated characters
    pattern = r'(.)\1{' + str(max_repeats) + ',}'
    
    def replace_repeats(match):
        char = match.group(1)
        return char * max_repeats
    
    return re.sub(pattern, replace_repeats, text)

def standardize_whitespace(text: str) -> str:
    """
    Standardize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with standardized whitespace
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip() 