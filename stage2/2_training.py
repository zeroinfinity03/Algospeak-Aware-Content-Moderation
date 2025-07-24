"""
🛡️ STAGE 2: STEP 2 - LLM FINE-TUNING MODULE

📋 EXECUTION FLOW (ML Training Pipeline):
1. 📊 1_dataset_prep.py prepares training data first
2. 🤖 THIS FILE (2_training.py) fine-tunes LLM on prepared data
3. 📈 3_evaluation.py analyzes the trained model performance
4. ⚡ 4_inference.py uses the trained model for production
5. 🚀 5_stage2_demo.py demonstrates the complete AI system

PURPOSE: Fine-tunes Llama 3.2 on normalized algospeak data for context-aware classification.
CALLED BY: Manual training process or automated ML pipeline
USES: Data from 1_dataset_prep.py
CREATES: Trained model used by 4_inference.py
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from transformers.trainer_utils import EvalPrediction
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb

from .dataset_prep import ContentModerationDataset, create_datasets_for_training


@dataclass
class TrainingConfig:
    """Configuration for content moderation fine-tuning."""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    num_labels: int = 4  # safe, potentially_harmful, harmful, extremely_harmful
    num_categories: int = 8  # none, hate_speech, self_harm, etc.
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data settings
    max_length: int = 512
    test_size: float = 0.2
    val_size: float = 0.1
    max_samples: Optional[int] = None
    
    # Normalization comparison
    compare_with_without_normalization: bool = True
    
    # Output settings
    output_dir: str = "models/content_moderation"
    logging_dir: str = "logs/content_moderation"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    
    # Logging
    logging_steps: int = 50
    use_wandb: bool = True
    wandb_project: str = "algospeak-content-moderation"
    
    # Early stopping  
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Hardware
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42


class ContentModerationTrainer:
    """Trainer for content moderation models with algospeak normalization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        set_seed(config.seed)
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
                name=f"content-moderation-{config.model_name.split('/')[-1]}"
            )
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Per-class metrics for harm detection
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1, 2, 3]
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_safe': precision_per_class[0] if len(precision_per_class) > 0 else 0,
            'precision_potentially_harmful': precision_per_class[1] if len(precision_per_class) > 1 else 0,
            'precision_harmful': precision_per_class[2] if len(precision_per_class) > 2 else 0,
            'precision_extremely_harmful': precision_per_class[3] if len(precision_per_class) > 3 else 0,
            'recall_safe': recall_per_class[0] if len(recall_per_class) > 0 else 0,
            'recall_potentially_harmful': recall_per_class[1] if len(recall_per_class) > 1 else 0,
            'recall_harmful': recall_per_class[2] if len(recall_per_class) > 2 else 0,
            'recall_extremely_harmful': recall_per_class[3] if len(recall_per_class) > 3 else 0,
        }
        
        return metrics
    
    def train_model_with_normalization(self) -> Dict[str, Any]:
        """Train model with algospeak normalization."""
        print("🚀 Training model WITH algospeak normalization...")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = create_datasets_for_training(
            tokenizer_name=self.config.model_name,
            max_samples=self.config.max_samples,
            normalize_algospeak=True
        )
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            problem_type="single_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}_with_normalization",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.logging_dir}_with_normalization",
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            warmup_steps=self.config.warmup_steps,
            report_to="wandb" if self.config.use_wandb else None,
            run_name="with_normalization"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            )]
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(f"{self.config.output_dir}_with_normalization")
        
        return {
            "train_results": train_result,
            "test_results": test_results,
            "model_path": f"{self.config.output_dir}_with_normalization"
        }
    
    def train_model_without_normalization(self) -> Dict[str, Any]:
        """Train model without algospeak normalization for comparison."""
        print("📊 Training model WITHOUT algospeak normalization for comparison...")
        
        # Create datasets without normalization
        train_dataset, val_dataset, test_dataset = create_datasets_for_training(
            tokenizer_name=self.config.model_name,
            max_samples=self.config.max_samples,
            normalize_algospeak=False
        )
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            problem_type="single_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}_without_normalization",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.logging_dir}_without_normalization",
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            warmup_steps=self.config.warmup_steps,
            report_to="wandb" if self.config.use_wandb else None,
            run_name="without_normalization"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            )]
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(f"{self.config.output_dir}_without_normalization")
        
        return {
            "train_results": train_result,
            "test_results": test_results,
            "model_path": f"{self.config.output_dir}_without_normalization"
        }
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline with comparison."""
        results = {}
        
        # Ensure output directories exist
        Path(self.config.output_dir).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.logging_dir).parent.mkdir(parents=True, exist_ok=True)
        
        # Train with normalization
        results["with_normalization"] = self.train_model_with_normalization()
        
        # Train without normalization for comparison
        if self.config.compare_with_without_normalization:
            results["without_normalization"] = self.train_model_without_normalization()
        
        # Compare results
        if "without_normalization" in results:
            results["comparison"] = self.compare_results(
                results["with_normalization"]["test_results"],
                results["without_normalization"]["test_results"]
            )
        
        # Save complete results
        results_file = Path(self.config.output_dir).parent / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Training complete! Results saved to {results_file}")
        return results
    
    def compare_results(self, with_norm: Dict, without_norm: Dict) -> Dict[str, float]:
        """Compare training results with and without normalization."""
        comparison = {}
        
        for metric in ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']:
            if metric in with_norm and metric in without_norm:
                improvement = with_norm[metric] - without_norm[metric]
                comparison[f"{metric}_improvement"] = improvement
                comparison[f"{metric}_improvement_percent"] = (improvement / without_norm[metric]) * 100
        
        print("\n🔍 NORMALIZATION IMPACT ANALYSIS")
        print("=" * 50)
        for metric, value in comparison.items():
            print(f"{metric}: {value:.4f}")
        
        return comparison


def train_content_moderation_model(
    model_name: str = "microsoft/DialoGPT-medium",
    max_samples: Optional[int] = None,
    use_wandb: bool = True
) -> Dict[str, Any]:
    """High-level function to train content moderation model."""
    
    config = TrainingConfig(
        model_name=model_name,
        max_samples=max_samples,
        use_wandb=use_wandb
    )
    
    trainer = ContentModerationTrainer(config)
    return trainer.run_full_training_pipeline()


if __name__ == "__main__":
    # Example usage
    results = train_content_moderation_model(
        model_name="microsoft/DialoGPT-medium",
        max_samples=1000,  # Small sample for testing
        use_wandb=False
    )
    
    print("Training completed!")
    print(f"Results: {results}") 