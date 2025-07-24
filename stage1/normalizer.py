"""
Wrapper file to import from 1_normalizer.py
"""
from pathlib import Path
import importlib.util

# Import from the actual numbered file
spec = importlib.util.spec_from_file_location("normalizer", Path(__file__).parent / "1_normalizer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Export the classes and functions
AlgospeakNormalizer = module.AlgospeakNormalizer
NormalizationResult = module.NormalizationResult
apply_character_substitution = module.apply_character_substitution
clean_repeated_characters = module.clean_repeated_characters
standardize_whitespace = module.standardize_whitespace 