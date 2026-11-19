import torch

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from models.base import LocalLALM


class Qwen2Audio(LocalLALM):
    NAME = "qwen2-audio-7b"
    SAMPLING_RATE = 16000

    def load(self, model_path, revision):
        self.processor = AutoProcessor.from_pretrained(model_path, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, revision=revision, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="eager",
        ).eval()

    def build_inputs(self, waveforms, questions):
        texts = []
        for question in questions:
            content = [{"type": "text", "text": question}, {"type": "audio", "audio": "placeholder"}]
            texts.append(
                self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )

        inputs = self.processor(
            text=texts,
            audio=waveforms,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.SAMPLING_RATE,
        )
        return inputs.to(self.model.device)
