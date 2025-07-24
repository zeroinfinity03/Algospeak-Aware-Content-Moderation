"""
Wrapper file to import from 2_detector.py
"""
from pathlib import Path
import importlib.util

# Import from the actual numbered file
spec = importlib.util.spec_from_file_location("detector", Path(__file__).parent / "2_detector.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Export the classes
AlgospeakDetector = module.AlgospeakDetector
AlgospeakMatch = module.AlgospeakMatch 