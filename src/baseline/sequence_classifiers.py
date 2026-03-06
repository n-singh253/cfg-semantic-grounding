"""
Unified Sequence Classifier Baseline for Defense.

This module provides a complete pipeline for training and using sequence classifiers
as defenses against adversarial prompts. It includes:
  1. Data Preparation - Extract and split training data from evaluation results
  2. Training - Fine-tune transformer models (BERT, RoBERTa, etc.)
  3. Evaluation - Use trained model as defense plugin

Usage:
    # Run complete pipeline (data prep + training)
    python -m src.baseline.sequence_classifiers.sequence_classifiers \\
        --config configs/baselines/sequence_classifiers.yaml
    
    # Use as defense plugin (automatic via runner)
    python -m src.runner --baseline-config configs/baselines/sequence_classifiers.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.baseline.base import BaseDefense
from src.baseline.common.data_preparation import prepare_training_data
from src.common.llm import LLMClient
from src.common.types import DefenseReturn


###### DATA LOADING AND DATASET ######

class PromptDataset(Dataset):
    """Dataset for prompt classification."""
    
    def __init__(self, instances: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        prompt = instance["prompt"]
        label = instance["label"]
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "instance_id": instance["instance_id"],
        }


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of instances."""
    instances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


###### TRAINING FUNCTIONS ######

def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:

    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    class_report = classification_report(
        all_labels,
        all_predictions,
        target_names=["benign", "malicious"],
        output_dict=True,
        zero_division=0,
    )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": avg_loss,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "predictions": all_predictions,
        "labels": all_labels,
        "logits": all_logits,
    }


def train_model(
    train_data_path: Path,
    test_data_path: Path,
    model_name: str,
    output_dir: Path,
    max_length: int = 512,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    random_seed: int = 42,
) -> None:
    """
    Train a sequence classification model for prompt safety detection.
    
    Args:
        train_data_path: Path to train.jsonl
        test_data_path: Path to test.jsonl
        model_name: HuggingFace model name (e.g., 'bert-base-uncased')
        output_dir: Directory to save trained model
        max_length: Maximum sequence length
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        warmup_ratio: Ratio of warmup steps
        random_seed: Random seed for reproducibility
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_instances = load_jsonl(train_data_path)
    test_instances = load_jsonl(test_data_path)
    
    train_labels = [inst["label"] for inst in train_instances]
    test_labels = [inst["label"] for inst in test_instances]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    model.to(device)
    
    train_dataset = PromptDataset(train_instances, tokenizer, max_length)
    test_dataset = PromptDataset(test_instances, tokenizer, max_length)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    best_f1 = 0
    training_history = []
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch,
        )
        
        test_metrics = evaluate(model, test_dataloader, device)
        
        epoch_time = time.time() - start_time
        
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "epoch_time": epoch_time,
        })
        
        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    best_model.to(device)
    
    final_metrics = evaluate(best_model, test_dataloader, device)
    
    # Save metrics and training history
    class_report = final_metrics["classification_report"]
    results = {
        "model_name": model_name,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_f1": best_f1,
        "final_metrics": {
            "accuracy": final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1": final_metrics["f1"],
            "confusion_matrix": final_metrics["confusion_matrix"],
        },
        "classification_report": class_report,
        "training_history": training_history,
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


###### DEFENSE PLUGIN ######

class SequenceClassifierDefense(BaseDefense):
    """
    Defense using a trained transformer-based sequence classifier.
    
    This defense loads a pre-trained model (e.g., BERT) and classifies
    the prompt as benign or malicious. It only considers the prompt,
    ignoring the patch.
    
    Config options:
        model_path: Path to trained model directory (required)
        max_length: Maximum sequence length (default: 512)
        threshold: Classification threshold (default: 0.5)
        device: Device to use ('cuda', 'cpu', or 'auto', default: 'auto')
        surrogate_mode: If True, use dummy classification (for testing)
    """
    
    name = "sequence_classifiers"
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: LLMClient,
        baseline_config_hash: str,
        run_root: Path,
        fidelity_mode: str,
    ) -> None:
        super().__init__(config, llm_client, baseline_config_hash, run_root, fidelity_mode)
        
        self.max_length = int(config.get("max_length", 512))
        self.threshold = float(config.get("threshold", 0.5))
        self.surrogate_mode = config.get("surrogate_mode", False)
        
        # Determine device
        device_config = config.get("device", "auto")
        if device_config == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)
        
        # Load model
        if not self.surrogate_mode:
            model_path = config.get("model_path")
            if not model_path:
                raise ValueError("model_path is required in config")
            
            model_path = Path(model_path)
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                raise
        else:
            self.tokenizer = None
            self.model = None
    
    def _classify_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Classify a prompt as benign or malicious.
        
        Args:
            prompt: The prompt text to classify
        
        Returns:
            Dict with classification results:
                - prediction: 0 (benign) or 1 (malicious)
                - score: Probability of malicious class
                - logits: Raw model logits
        """
        if self.surrogate_mode:

            score = 0.3 if len(prompt) < 500 else 0.7
            prediction = 1 if score >= self.threshold else 0
            return {
                "prediction": prediction,
                "score": score,
                "logits": [0.7, 0.3] if prediction == 0 else [0.3, 0.7],
                "surrogate": True,
            }
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        malicious_score = probs[0, 1].item()
        prediction = 1 if malicious_score >= self.threshold else 0
        
        return {
            "prediction": prediction,
            "score": malicious_score,
            "logits": logits[0].cpu().tolist(),
            "surrogate": False,
        }
    
    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ) -> DefenseReturn:
        """
        Evaluate whether to accept, reject, or edit a patch.
        
        This defense only uses the prompt for classification, ignoring
        the patch content.
        
        Args:
            prompt: The prompt/issue statement
            code_or_patch: The patch (ignored)
            all_tests: Test specifications (ignored)
            repo_code: Repository metadata (ignored)
        
        Returns:
            True (accept), False (reject), or str (edited prompt/patch)
        """

        result = self._classify_prompt(prompt)
        
        self.last_signals = {
            "prediction": result["prediction"],
            "malicious_score": result["score"],
            "benign_score": 1.0 - result["score"],
            "logits": result["logits"],
            "threshold": self.threshold,
            "max_length": self.max_length,
            "surrogate": result.get("surrogate", False),
            "prompt_length": len(prompt),
        }
        
        if result["prediction"] == 1:
            return False
        else:
            return True

###### MAIN PIPELINE ######

def run_pipeline(config: Dict[str, Any]) -> None:
    """
    Run the complete pipeline: data preparation → training.
    
    Args:
        config: Configuration dictionary loaded from YAML
    """
    # Step 1: Data Preparation
    data_prep_config = config.get("data_preparation", {})
    
    metadata = prepare_training_data(
        benign_results_path=data_prep_config["benign_results_path"],
        malicious_results_path=data_prep_config["malicious_results_path"],
        output_dir=data_prep_config["output_dir"],
        split_strategy=data_prep_config.get("split_strategy", "stratified_instance"),
        train_ratio=data_prep_config.get("train_ratio", 0.8),
        random_seed=data_prep_config.get("random_seed", 42),
        limit_per_class=data_prep_config.get("limit_per_class"),
    )
    
    # Step 2: Training
    training_config = config.get("training", {})
    prepared_data_dir = Path(data_prep_config["output_dir"])
    
    train_model(
        train_data_path=prepared_data_dir / "train.jsonl",
        test_data_path=prepared_data_dir / "test.jsonl",
        model_name=training_config["model_name"],
        output_dir=Path(training_config["output_dir"]),
        max_length=training_config.get("max_length", 512),
        epochs=training_config.get("epochs", 3),
        batch_size=training_config.get("batch_size", 16),
        learning_rate=training_config.get("learning_rate", 2e-5),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        random_seed=training_config.get("random_seed", 42),
    )


def main():
    """Main entry point for running the complete pipeline."""

    parser = argparse.ArgumentParser(
        description="Sequence Classifier Baseline - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file"
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    run_pipeline(config)

if __name__ == "__main__":
    main()
