

Comprehensive Fine-Tuning and Deployment Report

(For Qwen2.5-3B-Instruct, Algospeak Moderation System)

⸻

1. Types of Fine-Tuning for Large Language Models

A. Full Fine-Tuning

What it is:
	•	Updates all the model’s weights during training.
	•	Requires loading the entire model in FP16 or BF16 precision.

How it’s done:
	1.	Download the full base model (e.g., Llama 3.2, 7B or 13B).
	2.	Use PyTorch/Transformers with large GPUs (A100, H100).
	3.	Train with your dataset, updating billions of parameters.

Benefits:
	•	Fully adapts the model to your dataset.
	•	Best performance for large data if you have huge compute.

Drawbacks:
	•	Needs 50GB+ VRAM for 13B models.
	•	Takes days and costs thousands of dollars.
	•	Overkill for a MacBook Air and a classification task like algospeak moderation.

⸻

B. LoRA (Low-Rank Adaptation)

What it is:
	•	Keeps the base model weights frozen.
	•	Trains small low-rank adapter layers to capture new task knowledge.
	•	Adapters can be merged or used alongside the base model.

How it’s done:
	1.	Download the base model (FP16).
	2.	Load it in PyTorch (Apple MPS support exists, but it’s heavy).
	3.	Train only millions of parameters instead of billions.

Benefits:
	•	Much faster and cheaper than full fine-tuning.
	•	Adapters are tiny (MB-sized) and reusable for multiple tasks.

Drawbacks:
	•	Still must load full FP16 weights during training.
	•	On an 8GB RAM MacBook, even a 7B model can be hard to fit.
	•	Slower training because FP16 weights dominate memory.

⸻

C. QLoRA (Quantized LoRA)

What it is:
	•	Loads the base model in 4-bit quantization (NF4), reducing VRAM by 3–4×.
	•	Quantized weights stay frozen (not updated).
	•	Trains LoRA adapters in FP16 for quality.

How it’s done:
	1.	Download the base model (FP16).
	2.	Load it quantized (4-bit) with BitsAndBytes (Colab) or Unsloth (Mac).
	3.	Train LoRA adapters.
	4.	Save adapters, then merge and quantize for deployment.

Benefits:
	•	Huge memory savings:
	•	3B models fit in ~3GB VRAM.
	•	7B can fit in ~6GB with optimization.
	•	Can train on Colab GPUs or MacBook Air (8GB RAM).
	•	Produces high-quality adapters even on low-resource hardware.

Drawbacks:
	•	Cannot fine-tune the quantized base weights (only adapters).
	•	Slightly slower per-step than pure FP16 training, but far more efficient overall.

⸻

2. Why QLoRA is Our Only Practical Choice
	•	Our setup: MacBook Air (8GB RAM) for deployment, Colab (T4/A100/L4) for training.
	•	Full fine-tuning is impossible (VRAM limits).
	•	Standard LoRA is heavy (needs full FP16 weights in VRAM).
	•	QLoRA works everywhere:
	•	Loads the model in 4-bit NF4 quantization, slashing VRAM.
	•	Trains only LoRA adapters (tiny MB-sized).
	•	Fully compatible with Colab GPUs and Apple Silicon (M1).

⸻

3. How It Fits Our Project (Algospeak Moderation)

We are fine-tuning to:
	•	Detect algospeak (slang like “unalive”, “seggs”).
	•	Normalize text (so models see “kill”, “sex”) and classify it as harmful or safe.
	•	Output confidence scores and categories (self-harm, hate, adult content).

With QLoRA:
	•	The 3B reasoning model fits easily on Colab GPUs and Apple M1.
	•	Training is fast and memory-efficient.
	•	After training:
	•	We merge the adapters with base weights.
	•	We quantize to GGUF (4-bit) for ultra-fast inference with Ollama or llama.cpp.

⸻

4. Chosen Model: Qwen2.5-3B-Instruct

We use Qwen2.5-3B-Instruct because:
	•	Instruction-tuned reasoning model:
	•	Learns our task with 25–50% fewer steps than a base model.
	•	Can output short reasoning/explanations for flagged content (TrustLab-friendly).
	•	Same size as the base (3.09B parameters):
	•	~2.3 GB FP16.
	•	~1.2–1.5 GB RAM in 4-bit.
	•	Avoids extra compute needed to teach prompt-following (already trained for it).

Why Not the Base Model
	•	Base model is only for raw text prediction.
	•	Would need extra epochs to learn prompt formatting and structured outputs.
	•	Wastes Colab GPU time and resources.

⸻

5. Fine-Tuning Process (Step by Step)

Step 1 – Load Base Model (Quantized)
	•	Load Qwen2.5-3B-Instruct.
	•	Quantize in 4-bit NF4 using:
	•	BitsAndBytes (Colab) or
	•	Unsloth (Mac).
	•	Base weights remain frozen (not updated).

Step 2 – Attach LoRA Adapters
	•	Add trainable low-rank layers.
	•	Parameters (both Colab & Mac): r=16, alpha=32, dropout=0.05.
	•	Unsloth and Colab handle memory optimizations automatically.

Step 3 – Train on Algospeak Dataset
	•	Dataset is normalized and instruction-style:

Instruction: Classify the following text for harmful content.
Input: "I want to unalive myself"
Normalized: "I want to kill myself"
Response: Label: harmful, Category: self_harm, Severity: extreme


	•	Reasoning model needs 25–50% fewer epochs than a base model.

Step 4 – Save Outputs
	•	Fine-tuning produces only the LoRA adapters:

adapter_model.safetensors
adapter_config.json


	•	Save by:
	•	Direct files.download() from Colab,
	•	Or mount Google Drive,
	•	Or push to Hugging Face Hub.

⸻

6. Post-Training Steps (On Mac)
	1.	Download Adapters (from Colab or Hugging Face).
	2.	Download Base FP16 Model:

git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct


	3.	Merge LoRA Adapters with Base Weights → Hugging Face model:

config.json
tokenizer.json
model.safetensors


	4.	Quantize the merged model to GGUF (4-bit) using llama.cpp tools:

fine_tuned_model.gguf


	5.	Serve locally:
	•	Ollama:

ollama create algospeak -f ./Modelfile
ollama run algospeak


	•	llama.cpp / llama-cpp-python (integrated into FastAPI).

⸻

7. BitsAndBytes vs Unsloth (For Fine-Tuning)

BitsAndBytes (Colab)
	•	Strengths:
	•	Works with Hugging Face templates out of the box.
	•	Prebuilt Colab notebooks available (many models have ready scripts).
	•	Fast on T4, A100, L4 GPUs.
	•	Streams models from Hugging Face Hub (no manual downloads).
	•	Downsides:
	•	Not optimized for Apple Silicon (for local training).
	•	Session limits (12 hours).

Unsloth (Mac)
	•	Strengths:
	•	Optimized for Apple Silicon (M1/M2/M3) — much faster and leaner than BitsAndBytes locally.
	•	Handles 4-bit quantization automatically (no manual tuning).
	•	Useful if you want to continue training locally after Colab.
	•	Downsides:
	•	Less documentation, more manual steps to export.
	•	Slower than an A100 GPU if doing full fine-tunes.

⸻

8. Final Deliverables
	•	After Colab training:

adapter_model.safetensors
adapter_config.json


	•	After merging on Mac:

config.json
tokenizer.json
model.safetensors


	•	After quantization (final):

fine_tuned_model.gguf



This GGUF model is what you’ll run via Ollama or llama.cpp.

⸻

9. Folder Structure (Stage 3)

algospeak-moderation/
└── stage3/                          
    ├── dataset/                     
    │   └── algospeak_dataset.json
    ├── qlora_train_colab.ipynb      # Colab training (BitsAndBytes)
    ├── qlora_train_mac_unsloth.py   # Local training (Unsloth)
    ├── qlora_outputs/               
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── merge_adapters.py            
    ├── fine_tuned_model/            
    │   ├── config.json
    │   ├── tokenizer.json
    │   └── model.safetensors
    ├── quantize_to_gguf.py          
    ├── fine_tuned_model.gguf        
    └── README.md                    


⸻






