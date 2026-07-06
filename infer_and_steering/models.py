"""Unified loading, generation, and decoder-layer access for registry models.

Loader strategies (config.ModelSpec.loader):
    auto     transformers-native chat-template models (Qwen-VL, Pangea, Granite,
             SEA-LION VL, Aya Vision, InternVL3-hf, jina-vlm, ...)
    seq2seq  BLIP-2-style encoder-decoder models (mBLIP)
    minicpm  MiniCPM-V remote-code models with a custom .chat() API
    custom   not loadable from transformers alone (PALO, Maya) -> raises with
             a pointer to the upstream repo
"""

import logging

import torch
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
)

import config

log = logging.getLogger(__name__)

# Attribute paths where the language-model decoder stack hides, per architecture.
_LAYER_PATHS = (
    "language_model.model.layers",
    "model.language_model.layers",
    "language_model.layers",
    "llm.model.layers",
    "model.layers",
    "language_model.decoder.block",  # seq2seq (mT0)
)


def _resolve(module: torch.nn.Module, path: str) -> torch.nn.ModuleList | None:
    for attr in path.split("."):
        module = getattr(module, attr, None)
        if module is None:
            return None
    return module if isinstance(module, torch.nn.ModuleList) else None


class VLM:
    """One vision-language model behind generate(image, prompt) and decoder_layers()."""

    def __init__(self, name: str, dtype: torch.dtype = torch.bfloat16):
        self.name = name
        self.spec = config.MODELS[name]
        if self.spec.loader == "custom":
            raise NotImplementedError(f"{name}: {self.spec.note}")
        log.info("loading %s (%s)", name, self.spec.repo_id)
        kwargs = dict(
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=self.spec.trust_remote_code,
        )
        if self.spec.loader == "seq2seq":
            self.model = Blip2ForConditionalGeneration.from_pretrained(self.spec.repo_id, **kwargs)
            self.processor = AutoProcessor.from_pretrained(self.spec.repo_id)
        elif self.spec.loader == "minicpm":
            self.model = AutoModel.from_pretrained(self.spec.repo_id, **kwargs).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.spec.repo_id, trust_remote_code=True)
        else:
            self.model = self._load_auto(kwargs)
            self.processor = AutoProcessor.from_pretrained(
                self.spec.repo_id, trust_remote_code=self.spec.trust_remote_code
            )
        self.model.eval()

    def _load_auto(self, kwargs: dict):
        last_error = None
        for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
            try:
                return cls.from_pretrained(self.spec.repo_id, **kwargs)
            except ValueError as error:  # architecture not mapped to this auto-class
                last_error = error
        raise last_error

    @torch.inference_mode()
    def generate(self, image, prompt: str, max_new_tokens: int = 64) -> str:
        if self.spec.loader == "minicpm":
            return self.model.chat(
                msgs=[{"role": "user", "content": [image.convert("RGB"), prompt]}],
                tokenizer=self.tokenizer,
                sampling=False,
                max_new_tokens=max_new_tokens,
            )
        if self.spec.loader == "seq2seq":
            inputs = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt").to(
                self.model.device, self.model.dtype
            )
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image.convert("RGB")}, {"type": "text", "text": prompt}],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def decoder_layers(self) -> torch.nn.ModuleList:
        for path in _LAYER_PATHS:
            layers = _resolve(self.model, path)
            if layers is not None:
                return layers
        raise AttributeError(f"{self.name}: no decoder layer stack found; extend _LAYER_PATHS")
