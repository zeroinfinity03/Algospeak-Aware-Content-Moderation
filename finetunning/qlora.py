# NOTE: This is for collab where you get more gpu and more memory.

# step 1:

#!/usr/bin/env python3
"""
Complete Package Installation for QLoRA Fine-tuning
Run this first in your Colab notebook
"""

# Install required packages for QLoRA training
install_commands = [
    "pip install -q transformers>=4.45.0",
    "pip install -q peft>=0.7.0", 
    "pip install -q bitsandbytes>=0.42.0",
    "pip install -q accelerate>=0.25.0",
    "pip install -q datasets>=2.15.0",
    "pip install -q scipy>=1.11.0",
    "pip install -q huggingface_hub>=0.19.0"
]

print("🚀 Installing QLoRA Dependencies...")
for cmd in install_commands:
    print(f"Running: {cmd}")
    import subprocess
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {cmd.split()[2]} installed successfully")
    else:
        print(f"❌ Error: {result.stderr}")

print("\n🔐 HuggingFace Login Required:")
print("Run this after installation:")
print("from huggingface_hub import login")
print("login()")  # You'll enter your HF token here

print("\n✅ Installation complete! Ready for QLoRA training.")










#Step 2:
# HuggingFace Login (REQUIRED)
from huggingface_hub import login
login()  # You'll get a popup to enter your HF token
# WARNING: NEVER put your actual token in code! Use the login() popup instead









#Step 3:
#!/usr/bin/env python3
"""
Clean QLoRA Fine-tuning for Massive Algospeak Dataset
Optimized for L4 GPU with 79k+ samples
All fixes included: labels for tokenization, WandB documentation, no duplicates
"""

import torch
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import gc

# Configuration for massive dataset
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "qwen-algospeak-lora"
DATASET_FILE = "../dataset/training_dataset_colab.json"

print("🚀 Loading Massive Algospeak Dataset...")

# Memory-efficient dataset loading
def load_large_dataset(file_path):
    """Load large JSON dataset efficiently"""
    if not os.path.exists(file_path):
        print(f"❌ Dataset not found: {file_path}")
        print("📋 Please upload your training_dataset_colab.json file!")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    print(f"✅ Dataset loaded: {len(data):,} samples")
    
    # Check data quality
    algospeak_count = sum(1 for item in data if item.get('is_algospeak', False))
    print(f"📊 Algospeak samples: {algospeak_count:,} ({algospeak_count/len(data)*100:.1f}%)")
    
    return data

training_data = load_large_dataset(DATASET_FILE)
if not training_data:
    print("❌ Cannot proceed without valid training data")
    raise SystemExit("Please upload your dataset file")

# Load tokenizer
print("📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Optimized 4-bit quantization for L4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model with optimal settings
print("🤖 Loading Qwen2.5-3B with 4-bit quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="flash_attention_2",  # Try Flash Attention 2
    )
    print("✅ Flash Attention 2 enabled!")
except Exception as e:
    print(f"⚠️ Flash Attention 2 not available: {e}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
    )
    print("✅ Standard attention enabled")

# Prepare for efficient training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Optimized LoRA config for algospeak detection
lora_config = LoraConfig(
    r=16,                    # Good balance for accuracy
    lora_alpha=32,          # 2x rank for stability  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,      # Light dropout
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("🔧 Trainable parameters:")
model.print_trainable_parameters()

# Format for algospeak detection
def format_algospeak_prompt(sample):
    """Format with algospeak context"""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

# Process massive dataset efficiently
print("🔄 Processing massive dataset...")
formatted_data = []
for i, sample in enumerate(training_data):
    formatted_data.append({"text": format_algospeak_prompt(sample)})
    
    # Progress indicator for large dataset
    if (i + 1) % 10000 == 0:
        print(f"Processed {i+1:,}/{len(training_data):,} samples...")

dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"📚 Train: {len(dataset['train']):,} | Test: {len(dataset['test']):,}")

# Memory cleanup
del formatted_data, training_data
gc.collect()
torch.cuda.empty_cache()

# Efficient tokenization - FIXED VERSION
def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,  # Optimal for algospeak content
        return_tensors=None,
    )
    # For causal LM, labels = input_ids (this is critical!)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("🔤 Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=1000,  # Process in batches for efficiency
    remove_columns=dataset["train"].column_names
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# 📊 ABOUT WANDB (Weights & Biases) - EXPERIMENT TRACKING
"""
🤔 What is WandB?
WandB is an experiment tracking platform that creates beautiful dashboards for your training:
- 📈 Real-time loss curves and training graphs  
- 💾 Automatic model checkpoint management
- 📊 GPU usage, memory consumption tracking
- 🔄 Compare different training runs

🎯 Do You Need It?
❌ NO for basic training - You just want the model trained
✅ YES for research - You want detailed monitoring and graphs

🔧 How to Control WandB:
- ENABLE:  Add report_to="wandb" below (requires WandB account + API key)
- DISABLE: Add report_to="none" below (simple terminal output only)
- DEFAULT: No report_to parameter = WandB will try to initialize automatically

💡 RECOMMENDATION: Add report_to="none" for simple training without extra setup
"""

# L4-optimized training arguments  
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,      # Conservative for stability
    per_device_eval_batch_size=4,       # Larger for eval
    gradient_accumulation_steps=8,      # Effective batch = 16
    num_train_epochs=3,                 # Good for large dataset
    learning_rate=1e-4,                 # Optimal for QLoRA
    fp16=True,                          # Memory optimization
    logging_steps=100,                  # More frequent logging
    eval_strategy="steps",
    eval_steps=500,                     # Less frequent eval (large dataset)
    save_steps=1000,                    # Save checkpoints
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,        # Better for Colab
    gradient_checkpointing=True,        # Memory optimization
    warmup_steps=100,                   # Smooth training start
    # report_to="none",                 # 🔧 UNCOMMENT this line to disable WandB tracking
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

print("🚀 Starting QLoRA training on massive algospeak dataset...")
print(f"📊 Training on {len(tokenized_dataset['train']):,} samples")
print(f"🎯 Target: High-accuracy algospeak detection")

# Start training
trainer.train()

print("💾 Saving fine-tuned model...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 QLoRA Fine-tuning Complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}")

# Quick test of the fine-tuned model
print("\n🧪 Testing fine-tuned model...")
test_prompt = """### Instruction:
Analyze this content for harmfulness and classify it appropriately. Consider both direct language and coded language (algospeak).

### Input:
I want to unalive myself

### Response:
"""

# Quick inference test
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
test_response = response.split("### Response:")[-1].strip()
print(f"📝 Test Input: 'I want to unalive myself'")
print(f"🤖 Model Response: {test_response}")

print("\n✅ Ready for algospeak detection!")








# Step 4:

# Download adapters to your Mac
from google.colab import files
import zipfile
import subprocess

# Create zip of adapter files
subprocess.run(['zip', '-r', 'qwen_algospeak_adapters.zip', 'qwen-algospeak-lora/'])

# Download the zip file
files.download('qwen_algospeak_adapters.zip')










































# #!/usr/bin/env python3
# """
# Clean QLoRA Fine-tuning for Massive Algospeak Dataset
# Optimized for L4 GPU with 79k+ samples
# All fixes included: labels for tokenization, WandB documentation, no duplicates
# """

# # ============================================================================
# # 📦 IMPORTS AND DEPENDENCIES
# # ============================================================================

# import torch
# import json
# import os
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForSeq2Seq
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from datasets import Dataset
# import gc

# # ============================================================================
# # ⚙️ CONFIGURATION
# # ============================================================================

# # Configuration for massive dataset
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# OUTPUT_DIR = "qwen-algospeak-lora"
# DATASET_FILE = "training_dataset_colab.json"

# # ============================================================================
# # 📊 DATASET LOADING
# # ============================================================================

# print("🚀 Loading Massive Algospeak Dataset...")

# # Memory-efficient dataset loading
# def load_large_dataset(file_path):
#     """Load large JSON dataset efficiently"""
#     if not os.path.exists(file_path):
#         print(f"❌ Dataset not found: {file_path}")
#         print("📋 Please upload your training_dataset_colab.json file!")
#         return None
    
#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#     except Exception as e:
#         print(f"❌ Error loading dataset: {e}")
#         return None
    
#     print(f"✅ Dataset loaded: {len(data):,} samples")
    
#     # Check data quality
#     algospeak_count = sum(1 for item in data if item.get('is_algospeak', False))
#     print(f"📊 Algospeak samples: {algospeak_count:,} ({algospeak_count/len(data)*100:.1f}%)")
    
#     return data

# training_data = load_large_dataset(DATASET_FILE)
# if not training_data:
#     print("❌ Cannot proceed without valid training data")
#     raise SystemExit("Please upload your dataset file")

# # ============================================================================
# # 🔤 TOKENIZER SETUP
# # ============================================================================

# # Load tokenizer
# print("📝 Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # ============================================================================
# # 🤖 MODEL LOADING AND QUANTIZATION
# # ============================================================================

# # Optimized 4-bit quantization for L4
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# # Load model with optimal settings
# print("🤖 Loading Qwen2.5-3B with 4-bit quantization...")
# try:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True,
#         use_cache=False,
#         attn_implementation="flash_attention_2",  # Try Flash Attention 2
#     )
#     print("✅ Flash Attention 2 enabled!")
# except Exception as e:
#     print(f"⚠️ Flash Attention 2 not available: {e}")
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True,
#         use_cache=False,
#     )
#     print("✅ Standard attention enabled")

# # ============================================================================
# # 🔧 LORA CONFIGURATION
# # ============================================================================

# # Prepare for efficient training
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# # Optimized LoRA config for algospeak detection
# lora_config = LoraConfig(
#     r=16,                    # Good balance for accuracy
#     lora_alpha=32,          # 2x rank for stability  
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_dropout=0.05,      # Light dropout
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# model = get_peft_model(model, lora_config)
# print("🔧 Trainable parameters:")
# model.print_trainable_parameters()

# # ============================================================================
# # 📝 DATA PREPROCESSING
# # ============================================================================

# # Format for algospeak detection
# def format_algospeak_prompt(sample):
#     """Format with algospeak context"""
#     return f"""### Instruction:
# {sample['instruction']}

# ### Input:
# {sample['input']}

# ### Response:
# {sample['output']}"""

# # Process massive dataset efficiently
# print("🔄 Processing massive dataset...")
# formatted_data = []
# for i, sample in enumerate(training_data):
#     formatted_data.append({"text": format_algospeak_prompt(sample)})
    
#     # Progress indicator for large dataset
#     if (i + 1) % 10000 == 0:
#         print(f"Processed {i+1:,}/{len(training_data):,} samples...")

# dataset = Dataset.from_list(formatted_data)
# dataset = dataset.train_test_split(test_size=0.1, seed=42)

# print(f"📚 Train: {len(dataset['train']):,} | Test: {len(dataset['test']):,}")

# # Memory cleanup
# del formatted_data, training_data
# gc.collect()
# torch.cuda.empty_cache()

# # ============================================================================
# # 🔤 TOKENIZATION
# # ============================================================================

# # Efficient tokenization - FIXED VERSION
# def tokenize_function(examples):
#     # Tokenize the text
#     tokenized = tokenizer(
#         examples["text"],
#         truncation=True,
#         padding=False,
#         max_length=512,  # Optimal for algospeak content
#         return_tensors=None,
#     )
#     # For causal LM, labels = input_ids (this is critical!)
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

# print("🔤 Tokenizing dataset...")
# tokenized_dataset = dataset.map(
#     tokenize_function, 
#     batched=True,
#     batch_size=1000,  # Process in batches for efficiency
#     remove_columns=dataset["train"].column_names
# )

# # Data collator
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,
#     padding=True,
#     return_tensors="pt"
# )

# # ============================================================================
# # 📊 WANDB CONFIGURATION & TRAINING ARGUMENTS
# # ============================================================================

# # 📊 ABOUT WANDB (Weights & Biases) - EXPERIMENT TRACKING
# """
# 🤔 What is WandB?
# WandB is an experiment tracking platform that creates beautiful dashboards for your training:
# - 📈 Real-time loss curves and training graphs  
# - 💾 Automatic model checkpoint management
# - 📊 GPU usage, memory consumption tracking
# - 🔄 Compare different training runs

# 🎯 Do You Need It?
# ❌ NO for basic training - You just want the model trained
# ✅ YES for research - You want detailed monitoring and graphs

# 🔧 How to Control WandB:
# - ENABLE:  Add report_to="wandb" below (requires WandB account + API key)
# - DISABLE: Add report_to="none" below (simple terminal output only)
# - DEFAULT: No report_to parameter = WandB will try to initialize automatically

# 💡 RECOMMENDATION: Add report_to="none" for simple training without extra setup
# """

# # L4-optimized training arguments  
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=2,      # Conservative for stability
#     per_device_eval_batch_size=4,       # Larger for eval
#     gradient_accumulation_steps=8,      # Effective batch = 16
#     num_train_epochs=3,                 # Good for large dataset
#     learning_rate=1e-4,                 # Optimal for QLoRA
#     fp16=True,                          # Memory optimization
#     logging_steps=100,                  # More frequent logging
#     eval_strategy="steps",
#     eval_steps=500,                     # Less frequent eval (large dataset)
#     save_steps=1000,                    # Save checkpoints
#     save_total_limit=3,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     remove_unused_columns=False,
#     dataloader_pin_memory=False,        # Better for Colab
#     gradient_checkpointing=True,        # Memory optimization
#     warmup_steps=100,                   # Smooth training start
#     # report_to="none",                 # 🔧 UNCOMMENT this line to disable WandB tracking
# )

# # ============================================================================
# # 🚀 TRAINING EXECUTION
# # ============================================================================

# # Initialize trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["test"],
#     data_collator=data_collator,
# )

# print("🚀 Starting QLoRA training on massive algospeak dataset...")
# print(f"📊 Training on {len(tokenized_dataset['train']):,} samples")
# print(f"🎯 Target: High-accuracy algospeak detection")

# # Start training
# trainer.train()

# # ============================================================================
# # 💾 MODEL SAVING
# # ============================================================================

# print("💾 Saving fine-tuned model...")
# trainer.save_model()
# tokenizer.save_pretrained(OUTPUT_DIR)

# print("🎉 QLoRA Fine-tuning Complete!")
# print(f"📁 Model saved to: {OUTPUT_DIR}")

# # ============================================================================
# # 🧪 MODEL TESTING
# # ============================================================================

# # Quick test of the fine-tuned model
# print("\n🧪 Testing fine-tuned model...")
# test_prompt = """### Instruction:
# Analyze this content for harmfulness and classify it appropriately. Consider both direct language and coded language (algospeak).

# ### Input:
# I want to unalive myself

# ### Response:
# """

# # Quick inference test
# inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         temperature=0.7,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id
#     )

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# test_response = response.split("### Response:")[-1].strip()
# print(f"📝 Test Input: 'I want to unalive myself'")
# print(f"🤖 Model Response: {test_response}")

# print("\n✅ Ready for algospeak detection!")

# # ============================================================================
# # 🎯 END OF SCRIPT
# # ============================================================================ 