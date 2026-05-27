"""
Qwen2-VL / Qwen3-VL wrapper.
The orena-focus examples/inference.py also uses Qwen3-VL — good signal.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from .base import BaseVLM
from ..prompts.templates import build_prompt


class QwenVLModel(BaseVLM):

    def __init__(self, cfg: dict):
        self.model_name = cfg["model"]["name"]
        self.dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
        self.device = cfg["model"].get("device", "cuda")
        self.max_new_tokens = cfg["inference"].get("max_new_tokens", 64)
        self.temperature = cfg["inference"].get("temperature", 0.0)
        self.use_metadata = cfg["prompt"].get("use_metadata", True)
        self.system_prompt = cfg["prompt"].get("system", "")
        self.model = None
        self.processor = None

    def load(self) -> None:
        print(f"Loading {self.model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded.")

    def predict(self, request) -> str:
        assert self.model is not None, "Call load() before predict()"

        prompt_text = build_prompt(request, self.system_prompt, self.use_metadata)

        # Build messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": request.image},   # PIL Image
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[request.image],
            return_tensors="pt",
        ).to(self.device)

        do_sample = self.temperature > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
            )

        # Decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return answer.strip()
