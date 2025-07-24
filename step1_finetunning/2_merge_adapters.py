#!/usr/bin/env python3
"""
Step 2: Merge LoRA Adapters with Base Model
Combines your fine-tuned adapters with Qwen2.5-3B-Instruct
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from pathlib import Path

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "qlora_outputs/qwen-algospeak-lora"
OUTPUT_PATH = "merged_model"

def download_base_model():
    """Download base model if not already present"""
    print("📥 Downloading base Qwen2.5-3B-Instruct model...")
    
    # This will download and cache the model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ Base model loaded successfully")
    return tokenizer, model

def merge_adapters():
    """Merge LoRA adapters with base model"""
    print("🔄 Starting adapter merge process...")
    
    # Load base model
    print("📚 Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto", 
        trust_remote_code=True
    )
    
    # Load and merge adapters
    print("🔧 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("⚡ Merging adapters with base weights...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"💾 Saving merged model to {OUTPUT_PATH}/")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("🎉 Adapter merge complete!")
    return merged_model, tokenizer

def test_merged_model():
    """Quick test of the merged model"""
    print("🧪 Testing merged model...")
    
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Test algospeak detection
    test_prompt = """### Instruction:
Analyze this content for harmfulness and classify it appropriately. Consider both direct language and coded language (algospeak).

### Input:
I want to unalive myself

### Response:"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("📝 Test Response:")
    print(response.split("### Response:")[-1].strip())
    
    print("✅ Model test complete!")

def verify_merged_model():
    """Verify merged model files"""
    output_path = Path(OUTPUT_PATH)
    
    required_files = [
        "config.json",
        "model.safetensors", 
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    print("🔍 Verifying merged model files...")
    
    for file in required_files:
        file_path = output_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024*1024)  # MB
            print(f"  ✅ {file} ({size:.1f} MB)")
        else:
            print(f"  ❌ Missing: {file}")
            return False
    
    print("🎉 Merged model verified!")
    return True

if __name__ == "__main__":
    print("🚀 Step 2: Merge LoRA Adapters")
    
    # Check if adapters exist
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Adapter path not found: {ADAPTER_PATH}")
        print("Please run 1_download_adapters.py first")
        exit(1)
    
    try:
        # Merge adapters
        merged_model, tokenizer = merge_adapters()
        
        # Verify output
        if verify_merged_model():
            print("✅ Ready for GGUF conversion!")
            
            # Optional: Test the model
            test_merged_model()
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        print("Check your adapter files and try again") 