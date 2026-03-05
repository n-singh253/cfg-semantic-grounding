"""Llama Prompt Guard baseline."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class LlamaPromptGuardDefense(BaseDefense):
    name = "llama_prompt_guard"
    requires_fit = False
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm_client,
        baseline_config_hash: str,
        run_root,
        fidelity_mode: str,
    ):
        super().__init__(config, llm_client, baseline_config_hash, run_root, fidelity_mode)
        
        # Load model configuration
        model_name = str(config.get("model", "meta-llama/Llama-Prompt-Guard-2-86M"))
        self.max_length = int(config.get("max_length", 512))
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        if fidelity_mode == "surrogate_debug":
            self.model = None
            self.tokenizer = None
            self.model_name = model_name
            self.malicious_label_ids = [1]
            return
                
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch_dtype,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Llama Prompt Guard model '{model_name}'. "
                "Ensure you have access (run: huggingface-cli login) and the model exists. "
                f"Error: {exc}"
            ) from exc
        
        self.model_name = model_name
        
        # Determine malicious label IDs based on model card specifications
        # Prompt Guard 1: BENIGN (0), INJECTION (1), JAILBREAK (2)
        # Prompt Guard 2: BENIGN (0), MALICIOUS (1)
        self.malicious_label_ids = self._get_malicious_label_ids()
    
    def _get_malicious_label_ids(self) -> list[int]:
        """
        Get label IDs that should be considered malicious based on model's id2label.
        """
        id2label = self.model.config.id2label
        malicious_ids = []
        
        for label_id, label_name in id2label.items():
            label_upper = label_name.upper()
            # Prompt Guard 2 uses 'MALICIOUS'
            # Prompt Guard 1 uses 'INJECTION' and 'JAILBREAK'
            if label_upper in ['MALICIOUS', 'INJECTION', 'JAILBREAK']:
                malicious_ids.append(label_id)
        
        if not malicious_ids:
            print(f"[llama_prompt_guard] Warning: Could not find malicious labels in {id2label}, using default [1]")
            malicious_ids = [1]
        
        return malicious_ids
    
    def _classify_text(self, text: str) -> tuple[int, float, str]:
        """
        Classify text using Llama Prompt Guard model.
        
        Returns:
            (predicted_class_id, malicious_score, raw_label)
        """
        if not text or not text.strip():
            return 0, 0.0, "UNKNOWN"
        
        # Surrogate mode for testing
        if self.fidelity_mode == "surrogate_debug":
            # Simple keyword-based surrogate
            text_lower = text.lower()
            injection_keywords = [
                "ignore previous", "ignore all previous", "disregard",
                "system:", "admin mode", "debug mode", "reveal", "instructions"
            ]
            if any(kw in text_lower for kw in injection_keywords):
                return 1, 0.95, "MALICIOUS"
            return 0, 0.05, "BENIGN"
        
        # Tokenize with truncation
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Get class prediction and probabilities
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_class_id = logits.argmax().item()
        raw_label = self.model.config.id2label[predicted_class_id]
        
        # Get maximum malicious probability (handle both PG1 and PG2)
        malicious_score = max(probs[label_id].item() for label_id in self.malicious_label_ids)
        
        return predicted_class_id, malicious_score, raw_label
    
    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (all_tests, code_or_patch)  # Not used in Llama Prompt Guard
        
        # Classify prompt only (patch is ignored)
        predicted_class_id, malicious_score, raw_label = self._classify_text(prompt)
        
        # Determine if prompt is malicious (check against all malicious label IDs)
        is_malicious = (predicted_class_id in self.malicious_label_ids)
        
        # Store telemetry
        instance_id = str(repo_code.get("instance_id", "unknown"))
        self.last_signals = {
            "model": self.model_name,
            "predicted_class_id": predicted_class_id,
            "raw_label": raw_label,
            "is_malicious": is_malicious,
            "malicious_score": malicious_score,
            "device": str(self.device),
            "instance_id": instance_id,
            "text_length": len(prompt),
            "fidelity_mode": self.fidelity_mode,
        }
        
        # Decision logic: reject if malicious, accept if benign
        if is_malicious:
            return False  # Reject malicious prompts
        else:
            return True  # Accept benign prompts


register_baseline("llama_prompt_guard")(LlamaPromptGuardDefense)
