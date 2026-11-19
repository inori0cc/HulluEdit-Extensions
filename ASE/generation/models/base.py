"""Shared prompt utilities for the two audio model adapters."""

import torch


class LocalLALM:
    SAMPLING_RATE = 16000

    def __init__(self, model_path: str, revision: str | None):
        self.load(model_path, revision)

    def load(self, model_path: str, revision: str | None):
        raise NotImplementedError

    def build_inputs(self, waveforms, questions):
        raise NotImplementedError

    def extra_gen_kwargs(self) -> dict:
        return {}

    def prompt_len(self, inputs) -> int:
        return inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

    def decode(self, output_ids: torch.Tensor, inputs: dict) -> list[str]:
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [text.strip() for text in texts]
