"""
VLM Backend Registry — Lightweight Vision-Language Models

Supports:
    - Qwen2.5-VL-3B-Instruct
    - nv reasoncxr-3b   
    - MiniCPM-V3.5-VL-4B   

All models use 4-bit quantization by default (configurable via config.VLM_QUANTIZE).
Models are lazily loaded on first call and cached for subsequent use.

Usage:
    from src.generation.vlm_backends import get_vlm_backend

    backend = get_vlm_backend("qwen2.5-vl-3b")
    output  = backend.generate(prompt="...", image_paths=["/path/to/xray.jpg"])
"""

import gc
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Thread
from typing import Generator, List, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

# MiniCPM import — may not be installed everywhere
try:
    from transformers import MiniCPMV4_6ForConditionalGeneration
except ImportError:
    MiniCPMV4_6ForConditionalGeneration = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    VLM_MODELS,
    VLM_MAX_NEW_TOKENS,
    VLM_TEMPERATURE,
    VLM_QUANTIZE,
)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

_backend_cache: dict = {}


def clear_backend_cache():
    """Release cached VLM backend instances and clear CUDA memory."""
    global _backend_cache
    _backend_cache = {}
    cleanup_cuda()


# ─────────────────────────────────────────────
# QUANTIZATION HELPER
# ─────────────────────────────────────────────

def _get_quantization_config(quantize: str):
    """Return a BitsAndBytesConfig for the requested quantization level."""

    if quantize == "fp16":
        return None

    from transformers import BitsAndBytesConfig
    import torch

    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unknown quantization: {quantize}")


# VLM Backend Base
class VLMBackend:

    def generate(
        self,
        prompt,
        image_paths=None,
    ):
        raise NotImplementedError


class QwenBackend(VLMBackend):
    
    def __init__(
        self,
        model_name,
        hf_repo=None,
        quantize=VLM_QUANTIZE,
    ):

        cleanup_cuda()

        model_id = hf_repo or model_name
        quant_config = _get_quantization_config(quantize)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        base_model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs,
            )
        )

        self.model = base_model
        self.model.eval()

    def generate(
        self,
        prompt,
        image_paths=None,
    ):
    
        content = []
    
        # ========================================================
        # IMAGES
        # ========================================================
    
        if image_paths:
    
            for img in image_paths[:2]:
    
                content.append({
                    "type": "image",
                    "image": img,
                })
    
        # ========================================================
        # TEXT
        # ========================================================
    
        content.append({
            "type": "text",
            "text": prompt,
        })
    
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
    
        # ========================================================
        # CHAT TEMPLATE
        # ========================================================
    
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
        # ========================================================
        # PROCESS INPUTS
        # ========================================================
    
        inputs = self.processor(
            text=[text_prompt],
            images=[
                Image.open(p).convert("RGB").resize((128, 128))
                for p in (image_paths[:2] if image_paths else [])
            ] if image_paths else None,
            return_tensors="pt",
            max_pixels=224*224,
        )
    
        model_device = next(self.model.parameters()).device
        inputs = {
            k: v.to(model_device)
            for k, v in inputs.items()
        }
    
        # ========================================================
        # GENERATION
        # ========================================================
    
        with torch.inference_mode():
    
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
    
        # ========================================================
        # REMOVE PROMPT TOKENS
        # ========================================================
    
        generated_ids = outputs[
            :,
            inputs["input_ids"].shape[1]:
        ]
    
        # ========================================================
        # DECODE
        # ========================================================
    
        generated = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        del outputs
        del generated_ids
        del inputs
            
        return generated.strip()

    # ========================================================
    # STREAMING GENERATION (token-by-token)
    # ========================================================

    def generate_stream(
        self,
        prompt,
        image_paths=None,
        max_new_tokens=512,
    ) -> Generator[str, None, None]:
        """Yield tokens one-by-one for Streamlit streaming."""

        content = []

        if image_paths:
            for img in image_paths[:2]:
                content.append({
                    "type": "image",
                    "image": img,
                })

        content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[
                Image.open(p).convert("RGB").resize((128, 128))
                for p in (image_paths[:2] if image_paths else [])
            ] if image_paths else None,
            return_tensors="pt",
            max_pixels=224 * 224,
        )

        model_device = next(self.model.parameters()).device
        inputs = {
            k: v.to(model_device)
            for k, v in inputs.items()
        }

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
        }

        thread = Thread(
            target=self.model.generate,
            kwargs=gen_kwargs,
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

# class NVReasonCXRBackend:

#     def __init__(
#         self,
#         model_name,
#     ):

#         cleanup_cuda()

#         # ====================================================
#         # QUANTIZATION
#         # ====================================================

#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             llm_int8_enable_fp32_cpu_offload=True,
#         )


#         # ====================================================
#         # PROCESSOR
#         # ====================================================

#         self.processor = AutoProcessor.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#         )


#         # ====================================================
#         # MODEL
#         # ====================================================

#         base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             model_name,
#             quantization_config=bnb_config,
#             low_cpu_mem_usage=True,
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#         )

#         base_model = base_model.to(DEVICE)

#         self.model = base_model
#         self.model.eval()

#     # ========================================================
#     # GENERATION
#     # ========================================================

#     def generate(
#         self,
#         prompt,
#         image_paths=None,
#     ):
    
#         content = []
    
#         # ========================================================
#         # IMAGES
#         # ========================================================
    
#         if image_paths:
    
#             for img in image_paths[:2]:
    
#                 content.append({
#                     "type": "image",
#                     "image": img,
#                 })
    
#         # ========================================================
#         # TEXT
#         # ========================================================
    
#         content.append({
#             "type": "text",
#             "text": prompt,
#         })
    
#         messages = [
#             {
#                 "role": "user",
#                 "content": content,
#             }
#         ]
    
#         # ========================================================
#         # CHAT TEMPLATE
#         # ========================================================
    
#         text_prompt = self.processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
    
#         # ========================================================
#         # PROCESS INPUTS
#         # ========================================================
    
#         inputs = self.processor(
#             text=[text_prompt],
#             images=[
#                 Image.open(p).convert("RGB").resize((128, 128))
#                 for p in (image_paths[:2] if image_paths else [])
#             ] if image_paths else None,
#             return_tensors="pt",
#             max_pixels=224*224,
#         )
    
#         model_device = next(self.model.parameters()).device
#         inputs = {
#             k: v.to(model_device)
#             for k, v in inputs.items()
#         }
    
#         # ========================================================
#         # GENERATION
#         # ========================================================
    
#         with torch.inference_mode():
    
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=256,
#                 do_sample=False,
#             )
    
#         # ========================================================
#         # REMOVE PROMPT TOKENS
#         # ========================================================
    
#         generated_ids = outputs[
#             :,
#             inputs["input_ids"].shape[1]:
#         ]


#         # ========================================================
#         # DECODE
#         # ========================================================
    
#         generated = self.processor.batch_decode(
#             generated_ids,
#             skip_special_tokens=True,
#         )[0]

#         del outputs
#         del generated_ids
#         del inputs
        
#         cleanup_cuda()
    
#         return generated.strip()



# class MiniCPMBackend(VLMBackend):

#     def __init__(self, model_name):

#         cleanup_cuda()

#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#         )

#         self.processor = AutoProcessor.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#         )

#         self.model = (
#             MiniCPMV4_6ForConditionalGeneration
#             .from_pretrained(
#                 model_name,
#                 quantization_config=bnb_config,
#                 torch_dtype=torch.float16,
#                 trust_remote_code=True,
#                 device_map="auto",
#             )
#         )

#         self.model.eval()

#     # ========================================================
#     # GENERATION
#     # ========================================================

#     def generate(
#         self,
#         prompt,
#         image_paths=None,
#     ):

#         content = []

#         # ====================================================
#         # IMAGES
#         # ====================================================

#         if image_paths:

#             for img in image_paths[:2]:

#                 content.append({

#                     "type": "image",

#                     "image": img,
#                 })

#         # ====================================================
#         # TEXT
#         # ====================================================

#         content.append({

#             "type": "text",

#             "text": prompt,
#         })

#         messages = [

#             {
#                 "role": "user",

#                 "content": content,
#             }
#         ]

#         # ====================================================
#         # CHAT TEMPLATE
#         # ====================================================

#         text_prompt = self.processor.apply_chat_template(

#             messages,

#             tokenize=False,

#             add_generation_prompt=True,
#         )

#         # ====================================================
#         # LOAD IMAGES
#         # ====================================================

#         images = [

#             Image.open(p)
#             .convert("RGB")
#             .resize((128,128))

#             for p in image_paths[:2]

#         ] if image_paths else None

#         # ====================================================
#         # PROCESS INPUTS
#         # ====================================================

#         inputs = self.processor(

#             text=[text_prompt],

#             images=images,

#             return_tensors="pt",
#         )

#         model_device = next(
#             self.model.parameters()
#         ).device

#         inputs = {

#             k: v.to(model_device)

#             for k, v in inputs.items()
#         }

#         # ====================================================
#         # GENERATE
#         # ====================================================

#         with torch.inference_mode():

#             outputs = self.model.generate(

#                 **inputs,

#                 max_new_tokens=256,

#                 do_sample=False,
#             )

#         # ====================================================
#         # REMOVE PROMPT
#         # ====================================================

#         generated_ids = outputs[
#             :,
#             inputs["input_ids"].shape[1]:
#         ]

#         # ====================================================
#         # DECODE
#         # ====================================================

#         generated = self.processor.batch_decode(

#             generated_ids,

#             skip_special_tokens=True,
#         )[0]

#         cleanup_cuda()

#         return generated.strip()



# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

# Maps model short-name prefix → backend class
_BACKEND_MAP = {
    "qwen2.5": QwenBackend,
    # "nv-reasoncxr": NVReasonCXRBackend,
    # "minicpm":      MiniCPMBackend,
}


def get_vlm_backend(model_name: str, quantize: str = VLM_QUANTIZE) -> VLMBackend:
    """
    Get (or create + cache) a VLM backend by short name.

    Args:
        model_name: One of the keys in config.VLM_MODELS
                    (e.g. "qwen2.5-vl-3b", "llava-v1.5-7b")
        quantize:   "fp16", "8bit", or "4bit"

    Returns:
        A VLMBackend instance (lazily loaded).
    """

    cache_key = f"{model_name}_{quantize}"

    if cache_key in _backend_cache:
        return _backend_cache[cache_key]

    if model_name not in VLM_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(VLM_MODELS.keys())}"
        )

    hf_repo = VLM_MODELS[model_name]

    # Find matching backend class
    backend_cls = None
    for prefix, cls in _BACKEND_MAP.items():
        if model_name.startswith(prefix):
            backend_cls = cls
            break

    if backend_cls is None:
        raise ValueError(f"No backend registered for model '{model_name}'")

    backend = backend_cls(model_name, hf_repo, quantize)
    _backend_cache[cache_key] = backend

    return backend


def list_available_models() -> list:
    """Return list of available VLM model names."""
    return list(VLM_MODELS.keys())
