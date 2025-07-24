#!/usr/bin/env python3
"""
Step 4: Setup Ollama with Fine-tuned Model
Deploy GGUF model locally on MacBook Air
"""

import os
import subprocess
import json
from pathlib import Path

# Configuration
MODEL_NAME = "qwen-algospeak"
GGUF_MODEL_PATH = "qwen_algospeak_model_q4_k_m.gguf"  # Quantized version
MODELFILE_PATH = "Modelfile"

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Ollama not found")
    return False

def install_ollama():
    """Install Ollama on macOS"""
    print("📥 Installing Ollama...")
    print("Please install Ollama manually:")
    print("1. Visit: https://ollama.ai/download")
    print("2. Download for macOS")  
    print("3. Run the installer")
    print("4. Restart terminal")
    print()
    print("Or use Homebrew:")
    print("brew install ollama")
    print()
    input("Press Enter after installing Ollama...")

def create_modelfile():
    """Create Ollama Modelfile for the fine-tuned model"""
    print("📝 Creating Modelfile...")
    
    # Check if GGUF model exists
    if not Path(GGUF_MODEL_PATH).exists():
        print(f"❌ GGUF model not found: {GGUF_MODEL_PATH}")
        print("Please run 3_convert_to_gguf.py first")
        return False
    
    modelfile_content = f"""FROM ./{GGUF_MODEL_PATH}

# Model parameters optimized for algospeak detection
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt for algospeak moderation
SYSTEM \"\"\"You are an AI content moderator specialized in detecting harmful content and algospeak (coded language used to evade content filters).

Your task is to:
1. Analyze text for harmful content including coded language
2. Classify content as safe or harmful  
3. Identify the category (hate_speech, self_harm, adult_content, violence, general)
4. Assign severity (0=safe, 1=mild, 2=moderate, 3=severe, 4=extreme)
5. Detect if algospeak is being used

Always respond in this format:
Label: [safe/harmful]
Category: [category]
Severity: [0-4]
Algospeak: [true/false]
Explanation: [brief reasoning]
\"\"\"

# Template for consistent responses
TEMPLATE \"\"\"### Instruction:
Analyze this content for harmfulness and classify it appropriately. Consider both direct language and coded language (algospeak).

### Input:
{{ .Prompt }}

### Response:
\"\"\"
"""
    
    with open(MODELFILE_PATH, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile created: {MODELFILE_PATH}")
    return True

def create_ollama_model():
    """Create Ollama model from Modelfile"""
    print(f"🔨 Creating Ollama model: {MODEL_NAME}")
    
    cmd = ["ollama", "create", MODEL_NAME, "-f", MODELFILE_PATH]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Ollama model created successfully!")
        return True
    else:
        print(f"❌ Model creation failed: {result.stderr}")
        return False

def test_ollama_model():
    """Test the Ollama model"""
    print("🧪 Testing Ollama model...")
    
    test_cases = [
        "I want to unalive myself",  # Algospeak for suicide
        "This movie is amazing!",     # Safe content
        "Let's meet at the 🌽 tonight" # Potential coded language
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_input}")
        
        cmd = ["ollama", "run", MODEL_NAME, test_input]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("🤖 Response:")
            print(result.stdout.strip())
        else:
            print(f"❌ Test failed: {result.stderr}")
    
    print("\n✅ Model testing complete!")

def create_api_integration():
    """Create Python client for FastAPI integration"""
    print("🔗 Creating API integration code...")
    
    integration_code = '''#!/usr/bin/env python3
"""
Ollama Client for FastAPI Integration
Use this in your main.py to query the fine-tuned model
"""

import subprocess
import json
import asyncio
from typing import Dict, Any

class AlgospeakModerator:
    def __init__(self, model_name: str = "qwen-algospeak"):
        self.model_name = model_name
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content using fine-tuned Ollama model"""
        try:
            cmd = ["ollama", "run", self.model_name, text]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                return self._parse_response(response)
            else:
                return {
                    "error": f"Model error: {result.stderr}",
                    "label": "error",
                    "category": "general", 
                    "severity": 0,
                    "is_algospeak": False
                }
                
        except subprocess.TimeoutExpired:
            return {
                "error": "Model timeout",
                "label": "error", 
                "category": "general",
                "severity": 0,
                "is_algospeak": False
            }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured format"""
        result = {
            "label": "safe",
            "category": "general", 
            "severity": 0,
            "is_algospeak": False,
            "explanation": ""
        }
        
        for line in response.split('\\n'):
            line = line.strip()
            if line.startswith("Label:"):
                result["label"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Category:"):
                result["category"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Severity:"):
                try:
                    result["severity"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    result["severity"] = 0
            elif line.startswith("Algospeak:"):
                result["is_algospeak"] = line.split(":", 1)[1].strip().lower() == "true"
            elif line.startswith("Explanation:"):
                result["explanation"] = line.split(":", 1)[1].strip()
        
        return result

# Example usage
if __name__ == "__main__":
    moderator = AlgospeakModerator()
    
    test_text = "I want to unalive myself"
    result = moderator.analyze_content(test_text)
    
    print(f"Input: {test_text}")
    print(f"Result: {json.dumps(result, indent=2)}")
'''
    
    with open("ollama_client.py", 'w') as f:
        f.write(integration_code)
    
    print("✅ API integration code created: ollama_client.py")

def list_ollama_models():
    """List available Ollama models"""
    print("📋 Available Ollama models:")
    
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("❌ Failed to list models")

if __name__ == "__main__":
    print("🚀 Step 4: Setup Ollama Deployment")
    
    # Check Ollama installation
    if not check_ollama_installed():
        install_ollama()
        if not check_ollama_installed():
            print("❌ Please install Ollama and try again")
            exit(1)
    
    try:
        # Create Modelfile
        if create_modelfile():
            # Create Ollama model
            if create_ollama_model():
                print("🎉 Ollama setup complete!")
                
                # Test the model
                test_ollama_model()
                
                # Create API integration
                create_api_integration()
                
                # Show available models
                list_ollama_models()
                
                print("\\n✅ Ready for production!")
                print(f"Use: ollama run {MODEL_NAME}")
                print("Integration: Use ollama_client.py in your FastAPI")
    
    except KeyboardInterrupt:
        print("\\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"❌ Error during Ollama setup: {e}") 