#!/usr/bin/env python3
"""
Step 3: Convert Merged Model to GGUF Format
Optimizes model for llama.cpp / Ollama deployment on MacBook Air
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
MERGED_MODEL_PATH = "merged_model"
GGUF_OUTPUT_PATH = "qwen_algospeak_model.gguf"
LLAMA_CPP_PATH = "llama.cpp"  # Will be cloned if not exists

def install_requirements():
    """Install required packages for GGUF conversion"""
    print("📦 Installing GGUF conversion requirements...")
    
    requirements = [
        "torch",
        "transformers", 
        "sentencepiece",
        "protobuf"
    ]
    
    for package in requirements:
        try:
            __import__(package)
            print(f"  ✅ {package} already installed")
        except ImportError:
            print(f"  📥 Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

def clone_llama_cpp():
    """Clone and setup llama.cpp if not exists"""
    if Path(LLAMA_CPP_PATH).exists():
        print(f"✅ llama.cpp found at {LLAMA_CPP_PATH}")
        return
    
    print("📥 Cloning llama.cpp repository...")
    subprocess.run([
        "git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_PATH
    ], check=True)
    
    print("🔨 Building llama.cpp...")
    build_dir = Path(LLAMA_CPP_PATH)
    
    # Build for macOS (with Metal acceleration)
    subprocess.run([
        "make", "-C", str(build_dir), "-j", "8"
    ], check=True)
    
    print("✅ llama.cpp built successfully!")

def convert_to_gguf():
    """Convert merged model to GGUF format"""
    print("🔄 Converting merged model to GGUF...")
    
    # Check if merged model exists
    if not Path(MERGED_MODEL_PATH).exists():
        print(f"❌ Merged model not found at {MERGED_MODEL_PATH}")
        print("Please run 2_merge_adapters.py first")
        return False
    
    # Path to conversion script
    convert_script = Path(LLAMA_CPP_PATH) / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        print(f"❌ Conversion script not found: {convert_script}")
        return False
    
    print(f"⚡ Converting {MERGED_MODEL_PATH} → {GGUF_OUTPUT_PATH}")
    
    # Run conversion
    cmd = [
        sys.executable,
        str(convert_script),
        MERGED_MODEL_PATH,
        "--outfile", GGUF_OUTPUT_PATH,
        "--outtype", "f16"  # Use FP16 for good quality/size balance
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ GGUF conversion successful!")
        return True
    else:
        print(f"❌ Conversion failed: {result.stderr}")
        return False

def quantize_gguf(quantization="Q4_K_M"):
    """Quantize GGUF model for smaller size and faster inference"""
    print(f"🗜️  Quantizing model with {quantization}...")
    
    base_name = GGUF_OUTPUT_PATH.replace('.gguf', '')
    quantized_output = f"{base_name}_{quantization.lower()}.gguf"
    
    quantize_tool = Path(LLAMA_CPP_PATH) / "quantize"
    
    if not quantize_tool.exists():
        print(f"❌ Quantize tool not found: {quantize_tool}")
        return False
    
    cmd = [
        str(quantize_tool),
        GGUF_OUTPUT_PATH,
        quantized_output,
        quantization
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Quantized model saved: {quantized_output}")
        
        # Show file sizes
        original_size = Path(GGUF_OUTPUT_PATH).stat().st_size / (1024*1024*1024)
        quantized_size = Path(quantized_output).stat().st_size / (1024*1024*1024)
        
        print(f"📊 Original: {original_size:.2f} GB")
        print(f"📊 Quantized: {quantized_size:.2f} GB")
        print(f"📊 Compression: {(1-quantized_size/original_size)*100:.1f}%")
        
        return quantized_output
    else:
        print(f"❌ Quantization failed: {result.stderr}")
        return False

def test_gguf_model(model_path):
    """Test the GGUF model using llama.cpp"""
    print("🧪 Testing GGUF model...")
    
    main_tool = Path(LLAMA_CPP_PATH) / "main"
    
    if not main_tool.exists():
        print("⚠️  llama.cpp main tool not found, skipping test")
        return
    
    test_prompt = "### Instruction:\\nAnalyze this text for harmful content.\\n\\n### Input:\\nI want to unalive myself\\n\\n### Response:"
    
    cmd = [
        str(main_tool),
        "-m", model_path,
        "-p", test_prompt,
        "-n", "50",
        "--temp", "0.7"
    ]
    
    print("Running test inference...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print("📝 Test output:")
        print(result.stdout)
        print("✅ GGUF model test successful!")
    else:
        print(f"⚠️  Test failed: {result.stderr}")

def verify_gguf():
    """Verify GGUF file was created successfully"""
    if Path(GGUF_OUTPUT_PATH).exists():
        size = Path(GGUF_OUTPUT_PATH).stat().st_size / (1024*1024*1024)
        print(f"✅ GGUF model created: {GGUF_OUTPUT_PATH} ({size:.2f} GB)")
        return True
    else:
        print(f"❌ GGUF file not found: {GGUF_OUTPUT_PATH}")
        return False

if __name__ == "__main__":
    print("🚀 Step 3: Convert to GGUF Format")
    
    try:
        # Install requirements
        install_requirements()
        
        # Setup llama.cpp
        clone_llama_cpp()
        
        # Convert to GGUF
        if convert_to_gguf():
            print("🎉 GGUF conversion complete!")
            
            # Verify
            if verify_gguf():
                print("✅ Ready for Ollama deployment!")
                
                # Optional: Quantize for smaller size
                print("\n🤔 Quantize model for better performance? (y/n): ", end="")
                if input().lower().startswith('y'):
                    quantized_model = quantize_gguf("Q4_K_M")
                    if quantized_model:
                        print(f"🎯 Use this for Ollama: {quantized_model}")
                        # Test quantized model
                        test_gguf_model(quantized_model)
        
    except KeyboardInterrupt:
        print("\n⚠️  Conversion interrupted by user")
    except Exception as e:
        print(f"❌ Error during GGUF conversion: {e}")
        print("Check your merged model and try again") 