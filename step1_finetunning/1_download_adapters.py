#!/usr/bin/env python3
"""
Step 1: Download LoRA Adapters from Colab
Part of complete QLoRA → GGUF → Ollama workflow
"""

import os
import shutil
from pathlib import Path

def download_from_colab():
    """
    Instructions for downloading adapters from Colab
    """
    print("📥 Downloading LoRA Adapters from Colab")
    print("="*50)
    
    print("🔧 Option 1: Direct Download (Recommended)")
    print("""
# Add this to your Colab notebook AFTER training:

from google.colab import files
import zipfile
import os

# Create zip of adapter files
!zip -r qwen_algospeak_adapters.zip qwen-algospeak-lora/

# Download the zip file
files.download('qwen_algospeak_adapters.zip')
""")
    
    print("\n🔧 Option 2: Google Drive Mount")
    print("""
# Add this to your Colab notebook:

from google.colab import drive
drive.mount('/content/drive')

# Copy adapters to Drive
!cp -r qwen-algospeak-lora/ /content/drive/MyDrive/
""")
    
    print("\n🔧 Option 3: HuggingFace Hub Push")
    print("""
# Push to HF Hub (if you want to share publicly):

trainer.push_to_hub("your-username/qwen-algospeak-lora")
""")

def extract_adapters(zip_path="qwen_algospeak_adapters.zip"):
    """Extract downloaded adapters"""
    extract_dir = Path("qlora_outputs")
    extract_dir.mkdir(exist_ok=True)
    
    print(f"📂 Extracting {zip_path} to {extract_dir}/")
    
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("✅ Adapters extracted successfully!")
    
    # List contents
    for file in extract_dir.rglob("*"):
        if file.is_file():
            print(f"  📄 {file}")

def verify_adapters(adapter_dir="qlora_outputs/qwen-algospeak-lora"):
    """Verify adapter files are complete"""
    adapter_path = Path(adapter_dir)
    
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors", 
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    print("🔍 Verifying adapter files...")
    
    all_good = True
    for file in required_files:
        file_path = adapter_path / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024*1024)  # MB
            print(f"  ✅ {file} ({size:.1f} MB)")
        else:
            print(f"  ❌ Missing: {file}")
            all_good = False
    
    if all_good:
        print("🎉 All adapter files present and ready for merging!")
    else:
        print("⚠️  Some files missing. Check your Colab download.")
    
    return all_good

if __name__ == "__main__":
    print("🚀 Step 1: Download LoRA Adapters")
    print("Follow the instructions above to download from Colab")
    
    # If you have the zip file, uncomment:
    # extract_adapters()
    # verify_adapters() 