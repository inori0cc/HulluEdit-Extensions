from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from models import QWEN2_5_OMNI_SYSTEM_PROMPT
from models.base import LocalLALM


class Qwen2_5Omni(LocalLALM):
    NAME = "qwen2.5-omni-7b"
    SYSTEM_PROMPT = QWEN2_5_OMNI_SYSTEM_PROMPT
    SAMPLING_RATE = 16000

    def load(self, model_path, revision):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            revision=revision,
            device_map={"": 0},
            attn_implementation="sdpa",
        ).eval()

    def extra_gen_kwargs(self):
        # The standalone thinker loads without an eos token.
        return {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def build_inputs(self, waveforms, questions):
        texts = []
        for question in questions:
            content = [{"type": "text", "text": question}, {"type": "audio", "audio": "placeholder"}]
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
                },
                {"role": "user", "content": content},
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            )

        inputs = self.processor(
            text=texts,
            audio=waveforms,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.SAMPLING_RATE,
        )
        return inputs.to(self.model.device).to(self.model.dtype)
