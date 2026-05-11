"""Qwen generation service with CUDA-first Transformers inference."""

from __future__ import annotations

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .schemas import GenerationResult
from .settings import (
    HF_LOCAL_FILES_ONLY,
    QWEN_BATCH_SIZE,
    QWEN_ENFORCE_CUDA,
    QWEN_MAX_INPUT_TOKENS,
    QWEN_MAX_NEW_TOKENS,
    QWEN_MODEL_NAME,
    QWEN_TEMPERATURE,
    QWEN_TOP_P,
    QWEN_USE_FLASH_ATTENTION,
    QWEN_TORCH_DTYPE,
    QWEN_USE_8BIT_QUANTIZATION,
)

logger = logging.getLogger(__name__)


class QwenGenerationService:
    """Runs Qwen model inference with production-oriented defaults."""

    def __init__(self) -> None:
        has_cuda = torch.cuda.is_available()
        if QWEN_ENFORCE_CUDA and not has_cuda:
            raise RuntimeError("Qwen GPU mode requires CUDA but torch.cuda.is_available() is False")

        self.device = "cuda" if has_cuda else "cpu"
        self.dtype = self._pick_dtype()
        self.batch_size = QWEN_BATCH_SIZE

        preferred_attn_impl = "flash_attention_2" if QWEN_USE_FLASH_ATTENTION and self.device == "cuda" else "sdpa"

        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True,
            local_files_only=HF_LOCAL_FILES_ONLY,
        )
        
        # Prepare quantization config if enabled
        quantization_config = None
        if QWEN_USE_8BIT_QUANTIZATION and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Loading Qwen with 8-bit quantization to save memory")
        
        try:
            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": HF_LOCAL_FILES_ONLY,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True,
                "attn_implementation": preferred_attn_impl,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = self.dtype
            
            self.model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_NAME,
                **model_kwargs,
            )
        except Exception as exc:
            if preferred_attn_impl != "sdpa":
                logger.warning("Flash attention unavailable, fallback to SDPA: %s", exc)
                model_kwargs["attn_implementation"] = "sdpa"
                self.model = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL_NAME,
                    **model_kwargs,
                )
            else:
                raise

        if self.device != "cuda":
            self.model.to("cpu") # type: ignore

        self.model.eval()
        quant_info = " (8-bit quantized)" if quantization_config else f" | dtype={str(self.dtype)}"
        logger.info("Qwen model ready: %s | device=%s%s", QWEN_MODEL_NAME, self.device, quant_info)

    def generate(self, prompt: str) -> GenerationResult:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=QWEN_MAX_INPUT_TOKENS,
        )

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=QWEN_TEMPERATURE > 0,
                temperature=QWEN_TEMPERATURE,
                top_p=QWEN_TOP_P,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output[0][input_len:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return GenerationResult(
            answer=answer,
            prompt_tokens_estimate=int(input_len),
            completion_tokens_estimate=int(generated_ids.shape[-1]),
            model=QWEN_MODEL_NAME,
            used_cuda=self.device == "cuda",
            dtype=str(self.dtype),
        )

    @staticmethod
    def _pick_dtype() -> torch.dtype:
        """Pick dtype from settings or auto-detect."""
        dtype_str = QWEN_TORCH_DTYPE.lower()
        if dtype_str == "float16":
            return torch.float16
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        elif dtype_str == "float32":
            return torch.float32
        elif dtype_str == "auto":
            # Auto-detect
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        else:
            # Default fallback
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32
