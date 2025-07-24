"""
Wrapper file to import from 3_patterns.py
"""
from pathlib import Path
import importlib.util

# Import from the actual numbered file
spec = importlib.util.spec_from_file_location("patterns", Path(__file__).parent / "3_patterns.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Export all the patterns and functions
ALGOSPEAK_MAPPINGS = module.ALGOSPEAK_MAPPINGS
CHAR_SUBSTITUTIONS = module.CHAR_SUBSTITUTIONS
SYMBOL_PATTERNS = module.SYMBOL_PATTERNS
HOMOPHONE_MAPPINGS = module.HOMOPHONE_MAPPINGS
MISSPELLING_PATTERNS = module.MISSPELLING_PATTERNS
create_leet_patterns = module.create_leet_patterns
CONTEXTUAL_PATTERNS = module.CONTEXTUAL_PATTERNS
get_severity_level = module.get_severity_level 