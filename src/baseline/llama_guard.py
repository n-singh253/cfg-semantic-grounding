"""Llama Guard family baseline."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    import torch
    from transformers import AutoProcessor, Llama4ForConditionalGeneration
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    _IMPORT_ERROR = e

from src.baseline.base import BaseDefense
from src.baseline.registry import register_baseline


class LlamaGuardDefense(BaseDefense):
    name = "llama_guard"
    requires_fit = False
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm_client,
        baseline_config_hash: str,
        run_root,
        fidelity_mode: str,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                f"llama_guard baseline requires torch and Llama4ForConditionalGeneration. "
                f"Install with: pip install git+https://github.com/huggingface/transformers@v4.51.3-LlamaGuard-preview hf_xet. "
                f"Original error: {_IMPORT_ERROR}"
            )
        
        super().__init__(config, llm_client, baseline_config_hash, run_root, fidelity_mode)
                
        # Load model configuration
        model_name = str(config.get("model", "meta-llama/Llama-Guard-4-12B"))
        self.max_length = int(config.get("max_length", 4096))
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        if fidelity_mode == "surrogate_debug":
            self.model = None
            self.processor = None
            self.model_name = model_name
            return
                
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch_dtype,
            )
            # Llama4 requires attention_chunk_size to be set
            max_pos = getattr(self.model.config, "max_position_embeddings", 2048)
            default_chunk = min(2048, max_pos)
            if not hasattr(self.model.config, "attention_chunk_size") or self.model.config.attention_chunk_size is None:
                self.model.config.attention_chunk_size = default_chunk
            if hasattr(self.model, "language_model"):
                lm_cfg = getattr(self.model.language_model, "config", None)
                if lm_cfg is not None and (not hasattr(lm_cfg, "attention_chunk_size") or lm_cfg.attention_chunk_size is None):
                    lm_cfg.attention_chunk_size = default_chunk
            if hasattr(self.model.config, "text_config"):
                txt_cfg = self.model.config.text_config
                if txt_cfg is not None and (not hasattr(txt_cfg, "attention_chunk_size") or txt_cfg.attention_chunk_size is None):
                    txt_cfg.attention_chunk_size = default_chunk
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Llama Guard model '{model_name}'. "
                "Ensure you have access (run: huggingface-cli login) and the model exists. "
                f"Error: {exc}"
            ) from exc
        
        self.model_name = model_name
        print(f"[llama-guard] model-loaded | device={self.device}", flush=True)
    
    def _build_messages(self, user_message: str) -> list:
        """Build the messages list for apply_chat_template."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message}
                ],
            }
        ]
    
    def _classify_text(self, text: str) -> tuple[str, str, list[str]]:
        """
            Classify text.

            Returns:
            (verdict, raw_response, categories)
            verdict: "safe", "unsafe", or "unknown"
        """
        if not text or not text.strip():
            return "unknown", "", []
        
        if self.fidelity_mode == "surrogate_debug":
            text_lower = text.lower()
            unsafe_keywords = ["backdoor", "exfiltrate", "malicious", "exploit", "vulnerability"]
            if any(kw in text_lower for kw in unsafe_keywords):
                return "unsafe", "unsafe\nS1", ["S1"]
            return "safe", "safe", []
        
        messages = self._build_messages(text)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                use_cache=False,
            )
        
        # Decode response
        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )[0].strip()
        
        # Parse response
        verdict, categories = self._parse_response(response)
        return verdict, response, categories
    
    def _parse_response(self, response: str) -> tuple[str, list[str]]:
        """
        Parse Llama Guard response into verdict and categories.
        
        Expected format:
        Line 1: "safe" or "unsafe"
        Line 2 (if unsafe): Comma-separated category codes (e.g., "S1,S2")
        """
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if not lines:
            return "unknown", []
        
        first_line = lines[0].lower()
        if "unsafe" in first_line:
            verdict = "unsafe"
        elif "safe" in first_line:
            verdict = "safe"
        else:
            verdict = "unknown"
        
        categories = []
        if verdict == "unsafe" and len(lines) > 1:
            category_line = lines[1]
            categories = [cat.strip() for cat in category_line.split(",") if cat.strip()]
        
        return verdict, categories
    
    def defense(
        self,
        prompt: str,
        code_or_patch: str,
        all_tests: List[Any],
        repo_code: Dict[str, Any],
    ):
        _ = (all_tests, code_or_patch)
        
        verdict, raw_response, categories = self._classify_text(prompt)
        
        instance_id = str(repo_code.get("instance_id", "unknown"))
        self.last_signals = {
            "model": self.model_name,
            "verdict": verdict,
            "categories": categories,
            "raw_response": raw_response,
            "device": str(self.device),
            "instance_id": instance_id,
            "text_length": len(prompt),
            "fidelity_mode": self.fidelity_mode,
        }
        
        if verdict == "unsafe":
            return False
        elif verdict == "safe":
            return True
        else:
            return False


register_baseline("llama_guard")(LlamaGuardDefense)
