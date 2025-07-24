"""
🛡️ STAGE 2: STEP 3 - MODEL EVALUATION MODULE

📋 EXECUTION FLOW (ML Training Pipeline):
1. 📊 1_dataset_prep.py prepares training data first
2. 🤖 2_training.py fine-tunes LLM on prepared data
3. 📈 THIS FILE (3_evaluation.py) analyzes trained model performance
4. ⚡ 4_inference.py uses the validated model for production
5. 🚀 5_stage2_demo.py demonstrates the complete AI system

PURPOSE: Comprehensive evaluation measuring algospeak normalization impact on model performance.
CALLED BY: After model training to validate performance
ANALYZES: Models trained by 2_training.py
VALIDATES: Model readiness for 4_inference.py production use
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)
import pandas as pd

from .inference import ContentModerationInference, ModerationResult
from .dataset_prep import ContentSample, prepare_training_datasets


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Per-class metrics
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]  
    per_class_f1: Dict[str, float]
    
    # Confusion matrix
    confusion_matrix: np.ndarray
    
    # Business metrics
    false_positive_rate: float
    false_negative_rate: float
    harmful_content_recall: float  # Critical for safety
    
    # Algospeak specific
    algospeak_detection_rate: float
    normalization_impact: Dict[str, float]
    
    # Performance
    avg_processing_time_ms: float
    total_samples: int


class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Label mappings
        self.label_to_id = {
            "safe": 0,
            "potentially_harmful": 1,
            "harmful": 2, 
            "extremely_harmful": 3
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def evaluate_model(
        self,
        model: ContentModerationInference,
        test_samples: List[ContentSample],
        compare_with_baseline: bool = True
    ) -> EvaluationMetrics:
        """Evaluate model performance on test samples."""
        
        print(f"🔍 Evaluating model on {len(test_samples)} samples...")
        
        # Get predictions
        predictions = []
        ground_truth = []
        processing_times = []
        algospeak_detections = 0
        
        for sample in test_samples:
            result = model.predict_single(sample.text)
            predictions.append(result.label)
            ground_truth.append(sample.label)
            processing_times.append(result.processing_time_ms)
            
            if result.normalization_applied:
                algospeak_detections += 1
        
        # Convert to numeric labels
        pred_numeric = [self.label_to_id[p] for p in predictions]
        true_numeric = [self.label_to_id[t] for t in ground_truth]
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_numeric, pred_numeric)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_numeric, pred_numeric, average='weighted'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            true_numeric, pred_numeric, average=None, labels=[0, 1, 2, 3]
        )
        
        per_class_precision_dict = {
            self.id_to_label[i]: per_class_precision[i] if i < len(per_class_precision) else 0
            for i in range(4)
        }
        per_class_recall_dict = {
            self.id_to_label[i]: per_class_recall[i] if i < len(per_class_recall) else 0  
            for i in range(4)
        }
        per_class_f1_dict = {
            self.id_to_label[i]: per_class_f1[i] if i < len(per_class_f1) else 0
            for i in range(4)
        }
        
        # Confusion matrix
        cm = confusion_matrix(true_numeric, pred_numeric)
        
        # Business metrics
        total_samples = len(test_samples)
        false_positives = np.sum(cm) - np.trace(cm)  # Total misclassifications
        false_negatives = false_positives  # Simplified
        
        # Harmful content recall (critical for safety)
        harmful_indices = [i for i, label in enumerate(ground_truth) 
                          if label in ["harmful", "extremely_harmful"]]
        if harmful_indices:
            harmful_predictions = [predictions[i] for i in harmful_indices]
            harmful_correct = sum(1 for p in harmful_predictions 
                                if p in ["harmful", "extremely_harmful"])
            harmful_content_recall = harmful_correct / len(harmful_indices)
        else:
            harmful_content_recall = 0.0
        
        # Algospeak metrics
        algospeak_detection_rate = algospeak_detections / total_samples
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            per_class_precision=per_class_precision_dict,
            per_class_recall=per_class_recall_dict,
            per_class_f1=per_class_f1_dict,
            confusion_matrix=cm,
            false_positive_rate=false_positives / total_samples,
            false_negative_rate=false_negatives / total_samples,
            harmful_content_recall=harmful_content_recall,
            algospeak_detection_rate=algospeak_detection_rate,
            normalization_impact={},
            avg_processing_time_ms=np.mean(processing_times),
            total_samples=total_samples
        )
    
    def compare_with_without_normalization(
        self,
        model_with_norm_path: str,
        model_without_norm_path: str,
        test_samples: Optional[List[ContentSample]] = None,
        max_samples: int = 500
    ) -> Dict[str, Any]:
        """Compare model performance with and without normalization."""
        
        print("🔄 Comparing models with and without algospeak normalization...")
        
        # Load test samples if not provided
        if test_samples is None:
            _, _, test_samples = prepare_training_datasets(max_samples=max_samples)
        
        # Load both models
        model_with_norm = ContentModerationInference(model_with_norm_path)
        model_without_norm = ContentModerationInference(model_without_norm_path)
        
        # Evaluate both models
        metrics_with_norm = self.evaluate_model(model_with_norm, test_samples)
        metrics_without_norm = self.evaluate_model(model_without_norm, test_samples)
        
        # Calculate improvements
        improvements = {
            "accuracy_improvement": metrics_with_norm.accuracy - metrics_without_norm.accuracy,
            "precision_improvement": metrics_with_norm.precision - metrics_without_norm.precision,
            "recall_improvement": metrics_with_norm.recall - metrics_without_norm.recall,
            "f1_improvement": metrics_with_norm.f1_score - metrics_without_norm.f1_score,
            "harmful_recall_improvement": (
                metrics_with_norm.harmful_content_recall - 
                metrics_without_norm.harmful_content_recall
            ),
            "processing_time_impact": (
                metrics_with_norm.avg_processing_time_ms - 
                metrics_without_norm.avg_processing_time_ms
            )
        }
        
        # Calculate percentage improvements
        percentage_improvements = {}
        for metric, improvement in improvements.items():
            if metric.endswith("_improvement") and not metric.endswith("time_impact"):
                base_metric = metric.replace("_improvement", "")
                if base_metric == "harmful_recall":
                    base_value = metrics_without_norm.harmful_content_recall
                else:
                    base_value = getattr(metrics_without_norm, base_metric)
                
                if base_value > 0:
                    percentage_improvements[f"{metric}_percent"] = (improvement / base_value) * 100
        
        comparison_results = {
            "with_normalization": {
                "accuracy": metrics_with_norm.accuracy,
                "precision": metrics_with_norm.precision,
                "recall": metrics_with_norm.recall,
                "f1_score": metrics_with_norm.f1_score,
                "harmful_content_recall": metrics_with_norm.harmful_content_recall,
                "algospeak_detection_rate": metrics_with_norm.algospeak_detection_rate,
                "avg_processing_time_ms": metrics_with_norm.avg_processing_time_ms
            },
            "without_normalization": {
                "accuracy": metrics_without_norm.accuracy,
                "precision": metrics_without_norm.precision,
                "recall": metrics_without_norm.recall,
                "f1_score": metrics_without_norm.f1_score,
                "harmful_content_recall": metrics_without_norm.harmful_content_recall,
                "algospeak_detection_rate": metrics_without_norm.algospeak_detection_rate,
                "avg_processing_time_ms": metrics_without_norm.avg_processing_time_ms
            },
            "improvements": improvements,
            "percentage_improvements": percentage_improvements
        }
        
        # Save results
        results_file = self.output_dir / "normalization_comparison.json"
        with open(results_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Create visualizations
        self.create_comparison_visualizations(comparison_results)
        
        print(f"✅ Comparison complete! Results saved to {results_file}")
        return comparison_results
    
    def create_comparison_visualizations(self, comparison_results: Dict[str, Any]):
        """Create visualizations comparing model performance."""
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algospeak Normalization Impact Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        with_norm = comparison_results["with_normalization"]
        without_norm = comparison_results["without_normalization"]
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        with_values = [with_norm[m] for m in metrics]
        without_values = [without_norm[m] for m in metrics]
        
        # 1. Basic metrics comparison
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, without_values, width, label='Without Normalization', alpha=0.8)
        axes[0, 0].bar(x + width/2, with_values, width, label='With Normalization', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Improvement percentages
        improvements = comparison_results["percentage_improvements"]
        if improvements:
            improvement_metrics = list(improvements.keys())
            improvement_values = list(improvements.values())
            
            colors = ['green' if v > 0 else 'red' for v in improvement_values]
            axes[0, 1].barh(improvement_metrics, improvement_values, color=colors, alpha=0.7)
            axes[0, 1].set_xlabel('Improvement (%)')
            axes[0, 1].set_title('Percentage Improvements')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Harmful content recall (critical metric)
        harmful_metrics = ['Without Normalization', 'With Normalization']
        harmful_values = [without_norm['harmful_content_recall'], with_norm['harmful_content_recall']]
        
        axes[1, 0].bar(harmful_metrics, harmful_values, color=['red', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('Recall Rate')
        axes[1, 0].set_title('Harmful Content Detection (Critical Safety Metric)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(harmful_values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Processing time impact
        time_metrics = ['Without Normalization', 'With Normalization']
        time_values = [without_norm['avg_processing_time_ms'], with_norm['avg_processing_time_ms']]
        
        axes[1, 1].bar(time_metrics, time_values, color=['blue', 'orange'], alpha=0.7)
        axes[1, 1].set_ylabel('Processing Time (ms)')
        axes[1, 1].set_title('Processing Time Impact')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(time_values):
            axes[1, 1].text(i, v + 0.5, f'{v:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'normalization_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.output_dir / 'normalization_impact_analysis.png'}")
    
    def generate_evaluation_report(
        self,
        comparison_results: Dict[str, Any],
        model_info: Dict[str, str] = None
    ) -> str:
        """Generate comprehensive evaluation report."""
        
        report = []
        report.append("# 🛡️ ALGOSPEAK CONTENT MODERATION - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        if model_info:
            report.append("## 🤖 Model Information")
            for key, value in model_info.items():
                report.append(f"- **{key}**: {value}")
            report.append("")
        
        # Executive Summary
        with_norm = comparison_results["with_normalization"]
        without_norm = comparison_results["without_normalization"]
        improvements = comparison_results["improvements"]
        
        report.append("## 📊 Executive Summary")
        report.append(f"- **Accuracy Improvement**: {improvements['accuracy_improvement']:.3f} ({improvements.get('accuracy_improvement_percent', 0):.1f}%)")
        report.append(f"- **F1 Score Improvement**: {improvements['f1_improvement']:.3f} ({improvements.get('f1_improvement_percent', 0):.1f}%)")
        report.append(f"- **Harmful Content Recall**: {with_norm['harmful_content_recall']:.3f} (vs {without_norm['harmful_content_recall']:.3f})")
        report.append(f"- **Algospeak Detection Rate**: {with_norm['algospeak_detection_rate']:.3f}")
        report.append("")
        
        # Detailed Metrics
        report.append("## 📈 Detailed Performance Metrics")
        report.append("")
        report.append("### With Algospeak Normalization")
        report.append(f"- Accuracy: {with_norm['accuracy']:.4f}")
        report.append(f"- Precision: {with_norm['precision']:.4f}")
        report.append(f"- Recall: {with_norm['recall']:.4f}")
        report.append(f"- F1 Score: {with_norm['f1_score']:.4f}")
        report.append(f"- Processing Time: {with_norm['avg_processing_time_ms']:.1f}ms")
        report.append("")
        
        report.append("### Without Algospeak Normalization")
        report.append(f"- Accuracy: {without_norm['accuracy']:.4f}")
        report.append(f"- Precision: {without_norm['precision']:.4f}")
        report.append(f"- Recall: {without_norm['recall']:.4f}")
        report.append(f"- F1 Score: {without_norm['f1_score']:.4f}")
        report.append(f"- Processing Time: {without_norm['avg_processing_time_ms']:.1f}ms")
        report.append("")
        
        # Business Impact
        report.append("## 💰 Business Impact")
        report.append(f"- **Safety Improvement**: {improvements['harmful_recall_improvement']:.3f} better at catching harmful content")
        report.append(f"- **Processing Overhead**: {improvements['processing_time_impact']:.1f}ms additional time per classification")
        report.append(f"- **Algospeak Coverage**: {with_norm['algospeak_detection_rate']:.1%} of content contained detectable algospeak")
        report.append("")
        
        # Recommendations
        report.append("## 🎯 Recommendations")
        if improvements['f1_improvement'] > 0.05:
            report.append("- ✅ **RECOMMENDED**: Deploy model with algospeak normalization")
            report.append("- Significant improvement in content moderation accuracy")
        elif improvements['harmful_recall_improvement'] > 0.1:
            report.append("- ✅ **RECOMMENDED**: Deploy model with algospeak normalization")  
            report.append("- Critical improvement in harmful content detection")
        else:
            report.append("- ⚠️ **CONSIDER**: Limited improvement observed")
            report.append("- Evaluate cost-benefit of additional processing time")
        
        report.append("")
        report.append("---")
        report.append("*Report generated automatically by algospeak evaluation system*")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = self.output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"📋 Evaluation report saved to {report_file}")
        return report_text


def run_comprehensive_evaluation(
    model_with_norm_path: str,
    model_without_norm_path: str,
    max_samples: int = 500,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """Run comprehensive evaluation comparing models with/without normalization."""
    
    evaluator = ModelEvaluator(output_dir)
    
    # Run comparison
    results = evaluator.compare_with_without_normalization(
        model_with_norm_path=model_with_norm_path,
        model_without_norm_path=model_without_norm_path,
        max_samples=max_samples
    )
    
    # Generate report
    model_info = {
        "Model with Normalization": model_with_norm_path,
        "Model without Normalization": model_without_norm_path,
        "Test Samples": max_samples,
        "Evaluation Date": str(pd.Timestamp.now().date())
    }
    
    report = evaluator.generate_evaluation_report(results, model_info)
    
    print("🎉 Comprehensive evaluation complete!")
    return results


if __name__ == "__main__":
    # Example usage (will fail until models are trained)
    try:
        results = run_comprehensive_evaluation(
            model_with_norm_path="models/content_moderation_with_normalization",
            model_without_norm_path="models/content_moderation_without_normalization",
            max_samples=100
        )
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Models not found (expected until training is complete): {e}")
        print("Run training.py first to create the models!") 