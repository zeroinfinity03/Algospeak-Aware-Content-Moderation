"""
🛡️ STAGE 1: STEP 3 - ALGOSPEAK PATTERNS DATABASE MODULE

📋 EXECUTION FLOW:
1. 🚪 main.py (API) receives user text
2. ✨ 1_normalizer.py is called first
3. 🔍 2_detector.py is called by normalizer
4. 📊 THIS FILE (3_patterns.py) is used BY detector to load algospeak_patterns.json

PURPOSE: Loads 150+ research-backed algospeak patterns from JSON database.
CALLED BY: 2_detector.py (pattern detection engine)
LOADS: algospeak_patterns.json (the core pattern database file)
"""

import json
import re
import os
from typing import Dict, List, Tuple, Pattern, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PatternMetadata:
    """Metadata about the patterns database."""
    version: str
    source: str
    total_patterns: int
    coverage_improvement: str
    recall_improvement: str
    last_updated: str
    citations: List[str]

@dataclass
class BusinessMetrics:
    """Business impact metrics from research."""
    baseline_recall: float
    projected_recall: float
    improvement: float
    baseline_f1: float
    projected_f1: float
    coverage_before: float
    coverage_after: float
    false_positive_increase: float

# Get the path to the data directory
def get_data_path() -> Path:
    """Get the path to the data directory."""
    current_dir = Path(__file__).parent
    return current_dir  # JSON file is directly in stage1 folder

# Load patterns from JSON file
def load_patterns() -> Dict:
    """Load algospeak patterns from external JSON file."""
    data_file = get_data_path() / "algospeak_patterns.json"
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {data_file}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {data_file}")
        return {}

def load_metadata() -> Optional[PatternMetadata]:
    """Load pattern metadata."""
    patterns_data = load_patterns()
    metadata_dict = patterns_data.get("metadata", {})
    if not metadata_dict:
        return None
    
    return PatternMetadata(
        version=metadata_dict.get("version", "1.0.0"),
        source=metadata_dict.get("source", "Custom patterns"),
        total_patterns=metadata_dict.get("total_patterns", 0),
        coverage_improvement=metadata_dict.get("coverage_improvement", "Unknown"),
        recall_improvement=metadata_dict.get("recall_improvement", "Unknown"),
        last_updated=metadata_dict.get("last_updated", "Unknown"),
        citations=metadata_dict.get("citations", [])
    )

def load_business_metrics() -> Optional[BusinessMetrics]:
    """Load business impact metrics."""
    patterns_data = load_patterns()
    metrics_dict = patterns_data.get("business_metrics", {})
    if not metrics_dict:
        return None
    
    return BusinessMetrics(
        baseline_recall=metrics_dict.get("baseline_recall", 0.0),
        projected_recall=metrics_dict.get("projected_recall", 0.0),
        improvement=metrics_dict.get("improvement", 0.0),
        baseline_f1=metrics_dict.get("baseline_f1", 0.0),
        projected_f1=metrics_dict.get("projected_f1", 0.0),
        coverage_before=metrics_dict.get("coverage_before", 0.0),
        coverage_after=metrics_dict.get("coverage_after", 0.0),
        false_positive_increase=metrics_dict.get("false_positive_increase", 0.0)
    )

# Load data from external file
_patterns_data = load_patterns()
_metadata = load_metadata()
_business_metrics = load_business_metrics()

# Extract data from loaded JSON
ALGOSPEAK_MAPPINGS = _patterns_data.get("direct_mappings", {})
CHAR_SUBSTITUTIONS = _patterns_data.get("char_substitutions", {})
HOMOPHONE_MAPPINGS = _patterns_data.get("homophones", {})
MISSPELLING_PATTERNS = _patterns_data.get("misspellings", {})
LEETSPEAK_WORDS = _patterns_data.get("leetspeak", {})
EMOJI_SUBSTITUTIONS = _patterns_data.get("emoji_substitutions", {})
NUMERIC_CODES = _patterns_data.get("numeric_codes", {})
CONTEXTUAL_PATTERNS = _patterns_data.get("contextual_patterns", {})
CATEGORIES = _patterns_data.get("categories", {})
TRANSFORMATION_FAMILIES = _patterns_data.get("transformation_families", {})
CONFIDENCE_SCORING = _patterns_data.get("confidence_scoring", {})
SEVERITY_LEVELS = _patterns_data.get("severity_levels", {"mild": 1, "moderate": 2, "severe": 3, "extreme": 4})
SEVERITY_KEYWORDS = _patterns_data.get("severity_keywords", {})

# Create symbol patterns from loaded data
SYMBOL_PATTERNS = []
for pattern_data in _patterns_data.get("symbol_patterns", []):
    if len(pattern_data) == 2:
        SYMBOL_PATTERNS.append((pattern_data[0], pattern_data[1]))

def create_leet_patterns() -> List[Tuple[Pattern, str]]:
    """Create compiled regex patterns for leetspeak detection from loaded data.""" 
    patterns = []
    
    for leet, normal in LEETSPEAK_WORDS.items():
        pattern = re.compile(rf'\b{re.escape(leet)}\b', re.IGNORECASE)
        patterns.append((pattern, normal))
    
    return patterns

def create_emoji_patterns() -> List[Tuple[str, str]]:
    """Create emoji detection patterns."""
    return list(EMOJI_SUBSTITUTIONS.items())

def create_numeric_patterns() -> List[Tuple[Pattern, str]]:
    """Create numeric code detection patterns."""
    patterns = []
    
    for code, meaning in NUMERIC_CODES.items():
        pattern = re.compile(rf'\b{re.escape(code)}\b')
        patterns.append((pattern, meaning))
    
    return patterns

def get_pattern_by_category(category: str) -> List[str]:
    """Get all patterns in a specific category."""
    return CATEGORIES.get(category, [])

def get_transformation_families() -> Dict[str, str]:
    """Get information about the 7 transformation families."""
    return TRANSFORMATION_FAMILIES

def get_confidence_score(pattern_type: str) -> float:
    """Get confidence score for a pattern type."""
    return CONFIDENCE_SCORING.get(pattern_type, 0.5)

def get_all_patterns() -> Dict[str, any]:
    """Get all algospeak patterns organized by type."""
    return {
        'char_substitutions': CHAR_SUBSTITUTIONS,
        'direct_mappings': ALGOSPEAK_MAPPINGS,
        'symbol_patterns': SYMBOL_PATTERNS, 
        'homophones': HOMOPHONE_MAPPINGS,
        'misspellings': MISSPELLING_PATTERNS,
        'leetspeak_words': LEETSPEAK_WORDS,
        'emoji_substitutions': EMOJI_SUBSTITUTIONS,
        'numeric_codes': NUMERIC_CODES,
        'contextual': CONTEXTUAL_PATTERNS,
        'leet_patterns': create_leet_patterns(),
        'emoji_patterns': create_emoji_patterns(),
        'numeric_patterns': create_numeric_patterns(),
    }

def load_custom_patterns(file_path: str) -> Dict[str, str]:
    """Load custom algospeak patterns from external file."""
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                # Handle CSV or other formats here
                pass
    except FileNotFoundError:
        print(f"Warning: Could not find custom patterns file {file_path}")
        return {}

def get_severity_level(normalized_text: str) -> int:
    """Determine content severity level after normalization."""
    text_lower = normalized_text.lower()
    
    # Check each severity level (from highest to lowest)
    for level_name in ['extreme', 'severe', 'moderate', 'mild']:
        keywords = SEVERITY_KEYWORDS.get(level_name, [])
        if any(keyword in text_lower for keyword in keywords):
            return SEVERITY_LEVELS.get(level_name, 1)
    
    return SEVERITY_LEVELS.get('mild', 1)

def reload_patterns():
    """Reload patterns from data file (useful for updates)."""
    global _patterns_data, ALGOSPEAK_MAPPINGS, CHAR_SUBSTITUTIONS
    global HOMOPHONE_MAPPINGS, MISSPELLING_PATTERNS, LEETSPEAK_WORDS
    global EMOJI_SUBSTITUTIONS, NUMERIC_CODES, CONTEXTUAL_PATTERNS, CATEGORIES
    global TRANSFORMATION_FAMILIES, CONFIDENCE_SCORING, SEVERITY_LEVELS, SEVERITY_KEYWORDS, SYMBOL_PATTERNS
    global _metadata, _business_metrics
    
    _patterns_data = load_patterns()
    _metadata = load_metadata()
    _business_metrics = load_business_metrics()
    
    ALGOSPEAK_MAPPINGS = _patterns_data.get("direct_mappings", {})
    CHAR_SUBSTITUTIONS = _patterns_data.get("char_substitutions", {})
    HOMOPHONE_MAPPINGS = _patterns_data.get("homophones", {})
    MISSPELLING_PATTERNS = _patterns_data.get("misspellings", {})
    LEETSPEAK_WORDS = _patterns_data.get("leetspeak", {})
    EMOJI_SUBSTITUTIONS = _patterns_data.get("emoji_substitutions", {})
    NUMERIC_CODES = _patterns_data.get("numeric_codes", {})
    CONTEXTUAL_PATTERNS = _patterns_data.get("contextual_patterns", {})
    CATEGORIES = _patterns_data.get("categories", {})
    TRANSFORMATION_FAMILIES = _patterns_data.get("transformation_families", {})
    CONFIDENCE_SCORING = _patterns_data.get("confidence_scoring", {})
    SEVERITY_LEVELS = _patterns_data.get("severity_levels", {"mild": 1, "moderate": 2, "severe": 3, "extreme": 4})
    SEVERITY_KEYWORDS = _patterns_data.get("severity_keywords", {})
    
    # Recreate symbol patterns
    SYMBOL_PATTERNS = []
    for pattern_data in _patterns_data.get("symbol_patterns", []):
        if len(pattern_data) == 2:
            SYMBOL_PATTERNS.append((pattern_data[0], pattern_data[1]))

def get_pattern_stats() -> Dict[str, any]:
    """Get comprehensive statistics about loaded patterns."""
    total_direct = len(ALGOSPEAK_MAPPINGS)
    total_patterns = (
        total_direct + 
        len(HOMOPHONE_MAPPINGS) + 
        len(MISSPELLING_PATTERNS) + 
        len(LEETSPEAK_WORDS) +
        len(EMOJI_SUBSTITUTIONS) +
        len(NUMERIC_CODES)
    )
    
    return {
        'version': _metadata.version if _metadata else "Unknown",
        'total_patterns': total_patterns,
        'direct_mappings': len(ALGOSPEAK_MAPPINGS),
        'homophones': len(HOMOPHONE_MAPPINGS),
        'misspellings': len(MISSPELLING_PATTERNS),
        'leetspeak': len(LEETSPEAK_WORDS),
        'emoji_substitutions': len(EMOJI_SUBSTITUTIONS),
        'numeric_codes': len(NUMERIC_CODES),
        'char_substitutions': len(CHAR_SUBSTITUTIONS),
        'symbol_patterns': len(SYMBOL_PATTERNS),
        'contextual_patterns': len(CONTEXTUAL_PATTERNS),
        'categories': len(CATEGORIES),
        'transformation_families': len(TRANSFORMATION_FAMILIES),
        'research_backed': True,
        'coverage_improvement': _metadata.coverage_improvement if _metadata else "Unknown",
        'projected_recall': _business_metrics.projected_recall if _business_metrics else 0.0,
    }

def get_academic_citations() -> List[str]:
    """Get academic citations for the research."""
    if _metadata:
        return _metadata.citations
    return []

def get_business_impact() -> Dict[str, any]:
    """Get business impact metrics."""
    if not _business_metrics:
        return {}
    
    return {
        'baseline_recall': f"{_business_metrics.baseline_recall:.1%}",
        'projected_recall': f"{_business_metrics.projected_recall:.1%}",
        'improvement': f"+{_business_metrics.improvement:.1%}",
        'baseline_f1': f"{_business_metrics.baseline_f1:.2f}",
        'projected_f1': f"{_business_metrics.projected_f1:.2f}",
        'coverage_improvement': f"{_business_metrics.coverage_before:.0%} → {_business_metrics.coverage_after:.0%}",
        'false_positive_impact': f"+{_business_metrics.false_positive_increase:.1%}"
    }

def get_metadata() -> Optional[PatternMetadata]:
    """Get pattern metadata."""
    return _metadata 