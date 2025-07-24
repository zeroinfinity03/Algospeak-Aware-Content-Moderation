"""
🛡️ STAGE 2: STEP 1 - DATASET PREPARATION MODULE

📋 EXECUTION FLOW (ML Training Pipeline):
1. 📊 THIS FILE (1_dataset_prep.py) prepares training data FIRST
2. 🤖 2_training.py uses this data to fine-tune LLM
3. 📈 3_evaluation.py analyzes model performance
4. ⚡ 4_inference.py provides production classification
5. 🚀 5_stage2_demo.py demonstrates the complete AI system

PURPOSE: Prepares training datasets with algospeak augmentation for LLM fine-tuning.
CALLED BY: 2_training.py (for model training)
USES: Stage 1 normalization system for data augmentation
"""

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

# Add stage1 to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from normalizer import AlgospeakNormalizer


@dataclass
class ContentSample:
    """A content moderation training sample."""
    text: str
    label: str
    category: str
    severity: str
    confidence: float = 1.0
    is_algospeak_variant: bool = False
    original_text: Optional[str] = None


class ContentModerationDataset(TorchDataset):
    """PyTorch Dataset for content moderation training."""
    
    def __init__(
        self,
        samples: List[ContentSample],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        normalize_algospeak: bool = True
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize_algospeak = normalize_algospeak
        self.normalizer = AlgospeakNormalizer() if normalize_algospeak else None
        
        # Label mappings
        self.label_to_id = {
            "safe": 0,
            "potentially_harmful": 1, 
            "harmful": 2,
            "extremely_harmful": 3
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # Category mappings
        self.category_to_id = {
            "none": 0,
            "hate_speech": 1,
            "self_harm": 2,
            "adult_content": 3,
            "violence": 4,
            "harassment": 5,
            "misinformation": 6,
            "spam": 7
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample.text
        
        # Normalize algospeak if enabled
        if self.normalize_algospeak and self.normalizer:
            normalized = self.normalizer.normalize_text(text)
            text = normalized.normalized_text
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label_to_id[sample.label], dtype=torch.long),
            'category_labels': torch.tensor(self.category_to_id[sample.category], dtype=torch.long)
        }


def load_jigsaw_dataset(max_samples: Optional[int] = None) -> List[ContentSample]:
    """Load and process Jigsaw dataset from CSV files."""
    samples = []
    
    print("Loading Jigsaw dataset...")
    
    # Load training data
    train_df = pd.read_csv("dataset/train.csv")
    
    if max_samples:
        train_df = train_df.sample(n=min(max_samples, len(train_df)), random_state=42)
    
    print(f"Processing {len(train_df)} samples from Jigsaw dataset...")
    
    for _, row in train_df.iterrows():
        text = str(row['comment_text']).strip()
        if not text or len(text) < 10:  # Skip very short texts
            continue
            
        # Extract toxicity score
        target = float(row['target'])
        
        # Map to our label system
        if target >= 0.8:
            label = "extremely_harmful"
            severity = "extreme"
        elif target >= 0.6:
            label = "harmful"
            severity = "high"
        elif target >= 0.3:
            label = "potentially_harmful"
            severity = "medium"
        else:
            label = "safe"
            severity = "none"
        
        # Determine category based on available columns
        category = "none"
        if row.get('severe_toxicity', 0) > 0.5:
            category = "harassment"
        elif row.get('identity_attack', 0) > 0.5:
            category = "hate_speech"
        elif row.get('threat', 0) > 0.5:
            category = "violence"
        elif row.get('obscene', 0) > 0.5:
            category = "adult_content"
        elif row.get('insult', 0) > 0.5:
            category = "harassment"
        
        # Check for self-harm keywords
        text_lower = text.lower()
        if any(word in text_lower for word in ['suicide', 'kill myself', 'end my life', 'want to die']):
            category = "self_harm"
            if label == "safe":
                label = "harmful"
                severity = "high"
        
        samples.append(ContentSample(
            text=text,
            label=label,
            category=category,
            severity=severity,
            confidence=1.0
        ))
    
    print(f"Created {len(samples)} samples from Jigsaw dataset")
    return samples


def create_algospeak_variants(samples: List[ContentSample]) -> List[ContentSample]:
    """Create algospeak variants of existing training samples."""
    normalizer = AlgospeakNormalizer()
    variants = []
    
    # Load algospeak patterns
    algospeak_file = Path(__file__).parent.parent / "stage1" / "algospeak_patterns.json"
    try:
        with open(algospeak_file, 'r') as f:
            patterns = json.load(f)
    except Exception as e:
        print(f"Could not load algospeak patterns: {e}")
        return samples
    
    # Create reverse mapping (normal -> algospeak)
    reverse_mappings = {}
    for algospeak, normal in patterns.get("direct_mappings", {}).items():
        if normal not in reverse_mappings:
            reverse_mappings[normal] = []
        reverse_mappings[normal].append(algospeak)
    
    # Add homophones and other patterns
    for category in ["homophones", "misspellings", "leetspeak"]:
        for algospeak, normal in patterns.get(category, {}).items():
            if normal not in reverse_mappings:
                reverse_mappings[normal] = []
            reverse_mappings[normal].append(algospeak)
    
    print(f"Creating algospeak variants using {len(reverse_mappings)} patterns...")
    
    # Create variants for harmful samples
    for sample in samples:
        if sample.label in ["harmful", "extremely_harmful", "potentially_harmful"]:
            text_variants = []
            words = sample.text.lower().split()
            
            # Try to replace words with algospeak variants
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word in reverse_mappings:
                    # Create variant with algospeak substitution
                    for algospeak_variant in reverse_mappings[clean_word][:2]:  # Max 2 variants per word
                        new_words = words.copy()
                        new_words[i] = algospeak_variant
                        variant_text = ' '.join(new_words)
                        
                        variants.append(ContentSample(
                            text=variant_text,
                            label=sample.label,
                            category=sample.category,
                            severity=sample.severity,
                            is_algospeak_variant=True,
                            original_text=sample.text
                        ))
    
    print(f"Created {len(variants)} algospeak variants")
    all_samples = samples + variants
    random.shuffle(all_samples)
    return all_samples


def prepare_training_datasets(
    test_size: float = 0.2,
    val_size: float = 0.1,
    max_samples: Optional[int] = None,
    include_algospeak_variants: bool = True
) -> Tuple[List[ContentSample], List[ContentSample], List[ContentSample]]:
    """Prepare train/val/test datasets with optional algospeak augmentation."""
    
    # Load base samples
    samples = load_jigsaw_dataset(max_samples=max_samples)
    
    # Create algospeak variants
    if include_algospeak_variants:
        samples = create_algospeak_variants(samples)
    
    # Shuffle
    random.shuffle(samples)
    
    # Split datasets
    n_samples = len(samples)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples") 
    print(f"  Test: {len(test_samples)} samples")
    
    return train_samples, val_samples, test_samples


def save_dataset_for_colab(
    output_file: str = "training_dataset.json",
    max_samples: int = 10000
):
    """Save dataset in format ready for Google Colab fine-tuning."""
    
    print(f"🚀 Creating training dataset for Google Colab fine-tuning...")
    
    # Prepare datasets
    train_samples, val_samples, test_samples = prepare_training_datasets(
        max_samples=max_samples,
        include_algospeak_variants=True
    )
    
    # Convert to format suitable for instruction tuning
    def sample_to_instruction_format(sample: ContentSample) -> Dict[str, str]:
        """Convert sample to instruction-following format for LLM training."""
        
        instruction = "Classify the following text for harmful content and determine its category."
        input_text = f"Text: {sample.text}"
        
        # Create structured output
        output = f"Label: {sample.label}\nCategory: {sample.category}\nSeverity: {sample.severity}"
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "text": sample.text,
            "label": sample.label,
            "category": sample.category,
            "severity": sample.severity,
            "is_algospeak_variant": sample.is_algospeak_variant
        }
    
    # Convert all samples
    train_data = [sample_to_instruction_format(s) for s in train_samples]
    val_data = [sample_to_instruction_format(s) for s in val_samples]
    test_data = [sample_to_instruction_format(s) for s in test_samples]
    
    # Create final dataset
    dataset = {
        "metadata": {
            "total_samples": len(train_samples) + len(val_samples) + len(test_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "algospeak_variants": len([s for s in train_samples if s.is_algospeak_variant]),
            "created_for": "Qwen2.5-3B-Instruct fine-tuning on Google Colab"
        },
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }
    
    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dataset saved to {output_path}")
    print(f"📊 Total samples: {dataset['metadata']['total_samples']}")
    print(f"🎯 Algospeak variants: {dataset['metadata']['algospeak_variants']}")
    print(f"🚀 Ready for Google Colab fine-tuning!")
    
    return output_path


if __name__ == "__main__":
    print("🛡️ STAGE 2: DATASET PREPARATION FOR FINE-TUNING")
    print("=" * 60)
    
    # Create training dataset for Google Colab
    print("Creating training dataset with algospeak variants...")
    
    # Start with a manageable sample size
    dataset_path = save_dataset_for_colab(
        output_file="training_dataset.json",
        max_samples=10000  # Start with 10k samples
    )
    
    print(f"\n🎯 Next steps:")
    print(f"1. Upload {dataset_path} to Google Colab")
    print(f"2. Use the fine-tuning notebook to train Qwen2.5-3B-Instruct")
    print(f"3. The dataset includes algospeak variants for better performance!") 