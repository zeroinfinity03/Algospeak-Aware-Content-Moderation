"""
🛡️ STAGE 1: STEP 2 - ALGOSPEAK PATTERN DETECTION MODULE

📋 EXECUTION FLOW:
1. 🚪 main.py (API) receives user text
2. ✨ 1_normalizer.py is called first
3. 🔍 THIS FILE (2_detector.py) is called BY normalizer to find algospeak patterns
4. 📊 This file uses 3_patterns.py to load the pattern database (algospeak_patterns.json)

PURPOSE: THE BRAIN - Finds algospeak patterns with confidence scores across 7 transformation families.
CALLED BY: 1_normalizer.py (orchestrator)
CALLS: 3_patterns.py (to load pattern data)
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from patterns import (  # Import from 3_patterns.py
    ALGOSPEAK_MAPPINGS, CHAR_SUBSTITUTIONS, SYMBOL_PATTERNS,
    HOMOPHONE_MAPPINGS, MISSPELLING_PATTERNS, create_leet_patterns,
    CONTEXTUAL_PATTERNS, get_severity_level
)

@dataclass
class AlgospeakMatch:
    """Represents a detected algospeak pattern in text."""
    original: str       # Original algospeak text
    normalized: str     # Normalized canonical form
    start_pos: int     # Start position in text
    end_pos: int       # End position in text
    pattern_type: str  # Type of pattern matched
    confidence: float  # Confidence score (0.0 to 1.0)
    severity: int      # Severity level (1-4)

class AlgospeakDetector:
    """Main detector class for identifying algospeak patterns."""
    
    def __init__(self):
        """Initialize the detector with compiled patterns."""
        self.leet_patterns = create_leet_patterns()
        self.symbol_patterns = [(re.compile(pattern), replacement) 
                               for pattern, replacement in SYMBOL_PATTERNS]
        
        # Compile all mapping patterns for efficiency
        self.direct_patterns = self._compile_mapping_patterns(ALGOSPEAK_MAPPINGS)
        self.homophone_patterns = self._compile_mapping_patterns(HOMOPHONE_MAPPINGS)
        self.misspelling_patterns = self._compile_mapping_patterns(MISSPELLING_PATTERNS)
        
    def _compile_mapping_patterns(self, mappings: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
        """Compile mapping dictionaries into regex patterns."""
        patterns = []
        for algospeak, canonical in mappings.items():
            # Use word boundaries to avoid partial matches
            pattern = re.compile(rf'\b{re.escape(algospeak)}\b', re.IGNORECASE)
            patterns.append((pattern, canonical))
        return patterns
    
    def detect_patterns(self, text: str) -> List[AlgospeakMatch]:
        """
        Detect all algospeak patterns in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of AlgospeakMatch objects representing detected patterns
        """
        matches = []
        
        # Detect direct mappings (highest confidence)
        matches.extend(self._detect_direct_mappings(text))
        
        # Detect character substitutions (leetspeak)
        matches.extend(self._detect_leet_speak(text))
        
        # Detect symbol-based evasions
        matches.extend(self._detect_symbol_patterns(text))
        
        # Detect homophones and misspellings
        matches.extend(self._detect_homophones(text))
        matches.extend(self._detect_misspellings(text))
        
        # Detect contextual patterns (lower confidence)
        matches.extend(self._detect_contextual_patterns(text))
        
        # Remove overlapping matches (keep highest confidence)
        matches = self._resolve_overlaps(matches)
        
        return sorted(matches, key=lambda x: x.start_pos)
    
    def _detect_direct_mappings(self, text: str) -> List[AlgospeakMatch]:
        """Detect direct algospeak mappings."""
        matches = []
        
        for pattern, canonical in self.direct_patterns:
            for match in pattern.finditer(text):
                severity = get_severity_level(canonical)
                matches.append(AlgospeakMatch(
                    original=match.group(),
                    normalized=canonical,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    pattern_type='direct_mapping',
                    confidence=0.95,  # High confidence for direct mappings
                    severity=severity
                ))
        
        return matches
    
    def _detect_leet_speak(self, text: str) -> List[AlgospeakMatch]:
        """Detect leetspeak patterns."""
        matches = []
        
        for pattern, canonical in self.leet_patterns:
            for match in pattern.finditer(text):
                severity = get_severity_level(canonical)
                matches.append(AlgospeakMatch(
                    original=match.group(),
                    normalized=canonical,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    pattern_type='leetspeak',
                    confidence=0.85,
                    severity=severity
                ))
        
        return matches
    
    def _detect_symbol_patterns(self, text: str) -> List[AlgospeakMatch]:
        """Detect symbol-based evasion patterns."""
        matches = []
        
        for pattern, replacement in self.symbol_patterns:
            for match in pattern.finditer(text):
                original = match.group()
                normalized = pattern.sub(replacement, original)
                
                # Only consider it algospeak if the normalized form is different
                # and potentially problematic
                if original != normalized and self._is_potentially_harmful(normalized):
                    severity = get_severity_level(normalized)
                    matches.append(AlgospeakMatch(
                        original=original,
                        normalized=normalized,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        pattern_type='symbol_evasion',
                        confidence=0.70,  # Medium confidence
                        severity=severity
                    ))
        
        return matches
    
    def _detect_homophones(self, text: str) -> List[AlgospeakMatch]:
        """Detect homophone-based evasions."""
        matches = []
        
        for pattern, canonical in self.homophone_patterns:
            for match in pattern.finditer(text):
                severity = get_severity_level(canonical)
                matches.append(AlgospeakMatch(
                    original=match.group(),
                    normalized=canonical,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    pattern_type='homophone',
                    confidence=0.80,
                    severity=severity
                ))
        
        return matches
    
    def _detect_misspellings(self, text: str) -> List[AlgospeakMatch]:
        """Detect intentional misspellings."""
        matches = []
        
        for pattern, canonical in self.misspelling_patterns:
            for match in pattern.finditer(text):
                severity = get_severity_level(canonical)
                matches.append(AlgospeakMatch(
                    original=match.group(),
                    normalized=canonical,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    pattern_type='misspelling',
                    confidence=0.75,
                    severity=severity
                ))
        
        return matches
        
    def _detect_contextual_patterns(self, text: str) -> List[AlgospeakMatch]:
        """Detect contextual algospeak (requires surrounding context analysis)."""
        matches = []
        words = text.lower().split()
        
        for i, word in enumerate(words):
            if word in CONTEXTUAL_PATTERNS:
                # Analyze surrounding context for clues
                context_window = words[max(0, i-3):i+4]  # 3 words before/after
                
                for possible_meaning in CONTEXTUAL_PATTERNS[word]:
                    if self._has_contextual_clues(context_window, possible_meaning):
                        # Find the actual position in the original text
                        start_pos = text.lower().find(word)
                        if start_pos != -1:
                            end_pos = start_pos + len(word)
                            severity = get_severity_level(possible_meaning)
                            
                            matches.append(AlgospeakMatch(
                                original=word,
                                normalized=possible_meaning,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                pattern_type='contextual',
                                confidence=0.50,  # Lower confidence for contextual
                                severity=severity
                            ))
                            break  # Only take the first matching meaning
        
        return matches
    
    def _has_contextual_clues(self, context: List[str], meaning: str) -> bool:
        """Check if context provides clues for the intended meaning."""
        context_str = ' '.join(context)
        
        # Define context clues for different meanings
        clue_words = {
            'drugs': ['high', 'smoke', 'party', 'trip', 'dealer', 'stoned'],
            'adult_content': ['hot', 'sexy', 'bed', 'night', 'private', 'alone'],
            'sex': ['bed', 'night', 'together', 'private', 'intimate'],
            'violence': ['fight', 'hurt', 'pain', 'blood', 'angry', 'hate'],
            'fighting': ['match', 'ring', 'opponent', 'win', 'lose', 'bout']
        }
        
        if meaning in clue_words:
            return any(clue in context_str for clue in clue_words[meaning])
        
        return False
    
    def _is_potentially_harmful(self, text: str) -> bool:
        """Check if normalized text might be harmful content."""
        # Simple heuristic - check against common profanity/harmful terms
        harmful_indicators = [
            'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell',
            'kill', 'die', 'hate', 'violence', 'drug', 'sex'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in harmful_indicators)
    
    def _resolve_overlaps(self, matches: List[AlgospeakMatch]) -> List[AlgospeakMatch]:
        """Remove overlapping matches, keeping the one with highest confidence."""
        if not matches:
            return matches
        
        # Sort by position, then by confidence (descending)
        sorted_matches = sorted(matches, key=lambda x: (x.start_pos, -x.confidence))
        resolved = []
        
        for match in sorted_matches:
            # Check if this match overlaps with any already resolved match
            overlaps = False
            for resolved_match in resolved:
                if (match.start_pos < resolved_match.end_pos and 
                    match.end_pos > resolved_match.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                resolved.append(match)
        
        return resolved
    
    def get_detection_summary(self, matches: List[AlgospeakMatch]) -> Dict[str, any]:
        """Generate a summary of detection results."""
        if not matches:
            return {
                'total_matches': 0,
                'pattern_types': {},
                'severity_distribution': {},
                'avg_confidence': 0.0
            }
        
        pattern_counts = {}
        severity_counts = {}
        total_confidence = 0.0
        
        for match in matches:
            # Count pattern types
            pattern_counts[match.pattern_type] = pattern_counts.get(match.pattern_type, 0) + 1
            
            # Count severity levels
            severity_counts[match.severity] = severity_counts.get(match.severity, 0) + 1
            
            # Sum confidence for average
            total_confidence += match.confidence
        
        return {
            'total_matches': len(matches),
            'pattern_types': pattern_counts,
            'severity_distribution': severity_counts,
            'avg_confidence': total_confidence / len(matches),
            'highest_severity': max(match.severity for match in matches),
            'most_confident_match': max(matches, key=lambda x: x.confidence)
        } 