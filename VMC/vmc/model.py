"""Visual-only adaptation and native LLaVA prefill/decoding."""
import math
import re
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


LAYERS = tuple(range(12, 21))
RANK = 8
CAPTION_QUESTION = "Describe this image in detail."
YES_NO = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def prompt(question):
    return f"USER: <image>\n{question}\nASSISTANT:"


class VisualScope:
    def __init__(self):
        self.span = None

    @contextmanager
    def use(self, span):
        previous = self.span
        self.span = span
        try:
            yield
        finally:
            self.span = previous


class VisualMemory(nn.Module):
    """Apply the rank-r residual only to a single contiguous image span."""

    def __init__(self, base, scope, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scope = scope
        self.in_features = base.in_features
        self.out_features = base.out_features
        device = base.weight.device
        self.A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, hidden):
        native = self.base(hidden)
        if self.scope.span is None:
            return native
        start, end = self.scope.span
        if not 0 <= start < end <= hidden.shape[1]:
            raise ValueError("Image span is outside the current sequence.")
        residual = F.linear(F.linear(hidden[:, start:end].float(), self.A), self.B)
        adapted = native.clone()
        adapted[:, start:end] = native[:, start:end] + residual.to(native.dtype)
        return adapted


def attach(model, layers=LAYERS, rank=RANK):
    decoder = model.get_model()
    if getattr(model.config, "pretraining_tp", 1) != 1 or getattr(decoder, "gradient_checkpointing", False):
        raise ValueError("Visual masking requires unpartitioned K/V projections and gradient checkpointing off.")
    model.requires_grad_(False)
    scope = VisualScope()
    adapters = {}
    for layer in layers:
        attention = decoder.layers[layer].self_attn
        for name in ("k_proj", "v_proj"):
            adapter = VisualMemory(getattr(attention, name), scope, rank)
            setattr(attention, name, adapter)
            adapters[f"{layer}.{name}"] = adapter
    return scope, adapters


def adapter_state(adapters):
    return {
        key: {name: getattr(module, name).detach().cpu().clone() for name in ("A", "B")}
        for key, module in adapters.items()
    }


@torch.no_grad()
def restore(adapters, saved):
    if adapters.keys() != saved.keys():
        raise ValueError("Checkpoint layers do not match the VMC model.")
    for key, module in adapters.items():
        for name in ("A", "B"):
            param, value = getattr(module, name), saved[key][name]
            if value.shape != param.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid VMC checkpoint tensor {key}.{name}.")
    for key, module in adapters.items():
        for name in ("A", "B"):
            getattr(module, name).copy_(saved[key][name])


def load_adapter(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("scope") != "visual" or "adapters" not in checkpoint:
        raise ValueError("Expected a visual-only VMC adapter checkpoint.")
    return checkpoint


def contrastive_scores(target, reference, strength=0.5, threshold=0.1):
    if not 0 < threshold <= 1 or strength < 0 or not math.isfinite(strength):
        raise ValueError("Use strength >= 0 and 0 < threshold <= 1.")
    scores = target + strength * (target - reference)
    return scores.masked_fill(target < target.max() + math.log(threshold), -torch.inf)


def stable_answer(tokenizer, tokens):
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    match = YES_NO.match(text)
    return bool(match and (len(text) > match.end() or tokens[-1] == tokenizer.eos_token_id))


class NativeVMC:
    def __init__(self, model_path):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        tokenizer, model, processor, _ = load_pretrained_model(
            str(model_path), None, "llava-v1.5-7b", device="cuda", device_map={"": 0}
        )
        embeddings = model.get_input_embeddings().weight
        if model.config.hidden_size != 4096 or embeddings.dtype != torch.float16:
            raise ValueError("Expected the native FP16 LLaVA-1.5-7B model.")
        self.model = model.eval()
        self.backbone = model.get_model()
        self.tokenizer, self.processor = tokenizer, processor
        self.device, self.dtype = embeddings.device, embeddings.dtype
        self.scope, self.adapters = attach(model)
        self.image_token = IMAGE_TOKEN_INDEX
        self.encode_prompt = tokenizer_image_token
        self.process_images = process_images

    def image_ids(self, text):
        ids = self.encode_prompt(text, self.tokenizer, self.image_token, return_tensors="pt")
        return ids.unsqueeze(0).to(self.device)

    def image_position(self, ids):
        positions = (ids[0] == self.image_token).nonzero().flatten()
        if len(positions) != 1:
            raise ValueError("Use exactly one image placeholder per prompt.")
        return int(positions[0])

    @torch.no_grad()
    def pack(self, ids, features):
        start = self.image_position(ids)
        embed = self.model.get_input_embeddings()
        packed = torch.cat((embed(ids[:, :start]), features, embed(ids[:, start + 1:])), dim=1)
        return packed, (start, start + features.shape[1])

    @torch.no_grad()
    def image_features(self, image):
        ids = self.image_ids(prompt(CAPTION_QUESTION))
        pixels = self.process_images([image], self.processor, self.model.config)
        if isinstance(pixels, list):
            pixels = [part.to(device=self.device, dtype=self.dtype) for part in pixels]
        else:
            pixels = pixels.to(device=self.device, dtype=self.dtype)
        _, position_ids, attention_mask, _, embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, pixels
        )
        if embeds is None or embeds.shape[1] - ids.shape[1] + 1 != 576:
            raise ValueError("Expected 576 projected image tokens. Check the native LLaVA version.")
        start = self.image_position(ids)
        features = embeds[:, start:start + 576].detach().clone()
        reconstructed, _ = self.pack(ids, features)
        if not torch.equal(reconstructed, embeds):
            raise ValueError("Release prompt packing differs from native LLaVA packing.")
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError("Expected an unpadded single-image prompt.")
        if position_ids is not None:
            expected = torch.arange(embeds.shape[1], device=self.device)
            if not torch.equal(position_ids.flatten(), expected):
                raise ValueError("Expected contiguous position IDs.")
        return features

    def training_tail(self, features, prefix, completion):
        text = prompt(CAPTION_QUESTION)
        if prefix.strip():
            text += " " + prefix.rstrip()
        prefix_ids = self.image_ids(text)
        target = self.tokenizer(" " + completion, add_special_tokens=False)["input_ids"]
        if not 1 <= len(target) <= 96:
            raise ValueError("Expected 1 to 96 supervised tokens.")
        targets = torch.tensor([target], device=self.device, dtype=prefix_ids.dtype)
        embeds, span = self.pack(torch.cat((prefix_ids, targets), dim=1), features)
        # Logits one position before each supervised token predict that token.
        first = embeds.shape[1] - len(target) - 1
        with self.scope.use(span):
            output = self.backbone(inputs_embeds=embeds, use_cache=False, return_dict=True)
        logits = self.model.lm_head(output.last_hidden_state[0, first:first + len(target)]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite teacher-forcing logits.")
        return logits, targets[0]

    def caption_loss(self, features, caption):
        logits, targets = self.training_tail(features, "", caption)
        return F.cross_entropy(logits, targets)

    def noun_score(self, features, prefix, noun):
        logits, targets = self.training_tail(features, prefix, noun)
        return -F.cross_entropy(logits, targets)

    def decode_step(self, ids, features, last_token, past, adapted):
        if past is None:
            if features is None:
                with self.scope.use(None):
                    result = self.backbone(input_ids=ids, use_cache=True, return_dict=True)
            else:
                embeds, span = self.pack(ids, features)
                with self.scope.use(span if adapted else None):
                    result = self.backbone(inputs_embeds=embeds, use_cache=True, return_dict=True)
        else:
            token = torch.tensor([[last_token]], device=self.device, dtype=ids.dtype)
            # New positions in cached decoding are all text positions.
            with self.scope.use(None):
                result = self.backbone(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        logits = self.model.lm_head(result.last_hidden_state[0, -1][None]).float()[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite next-token logits.")
        return logits, result.past_key_values

    @torch.inference_mode()
    def generate(
        self, image, question=CAPTION_QUESTION, max_tokens=128, reference=None,
        adapted=True, early_answer=False, text_reference=False,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        ids = self.image_ids(prompt(question))
        target_features = self.image_features(image)
        reference_features = self.image_features(reference) if reference is not None else None
        if text_reference and reference is not None:
            raise ValueError("Choose one reference type.")
        ref_ids = ids[:, ids[0] != self.image_token] if text_reference else ids
        target_cache = reference_cache = None
        tokens = []
        for _ in range(max_tokens):
            last = tokens[-1] if tokens else None
            logits, target_cache = self.decode_step(ids, target_features, last, target_cache, adapted)
            if reference_features is not None or text_reference:
                ref_logits, reference_cache = self.decode_step(
                    ref_ids, reference_features, last, reference_cache, adapted and not text_reference,
                )
                logits = contrastive_scores(logits, ref_logits)
            tokens.append(int(logits.argmax()))
            if tokens[-1] == self.tokenizer.eos_token_id or (early_answer and stable_answer(self.tokenizer, tokens)):
                break
        return {"text": self.tokenizer.decode(tokens, skip_special_tokens=True), "tokens": tokens}
