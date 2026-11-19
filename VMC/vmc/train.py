"""Train the caption and absent-object stages with resumable adapter checkpoints."""
import fcntl
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from .data import load_json, sha256_file, validate_images
from .model import LAYERS, RANK, NativeVMC, adapter_state, load_adapter, restore


LR = 2e-5
WEIGHT_DECAY = 0.01
NOUN_WEIGHT = 0.1


def negative_loss(score, reference):
    # The original run used beta=1 and scores averaged over the noun's tokens.
    return 2 * F.softplus(score - reference)


def caption_order(size, epochs=2):
    order = []
    for epoch in range(epochs):
        indices = list(range(size))
        random.Random(42 + epoch).shuffle(indices)
        order.extend(indices)
    return order


def correction_order(size, updates=64, accumulation=4):
    rng = random.Random(53)
    order = []
    while len(order) < updates * accumulation:
        indices = list(range(size))
        rng.shuffle(indices)
        order.extend(indices)
    return order[:updates * accumulation]


@contextmanager
def training_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".train.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another training process is using {path}.") from error
        yield


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def features(engine, root, row):
    with Image.open(root / row["image"]) as image:
        return engine.image_features(image.convert("RGB"))


@torch.no_grad()
def score_reference(engine, examples, image_dir, output, signature):
    path = output / "reference.pt"
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError("Reference scores do not match this run.")
        return checkpoint["scores"]
    scores = []
    for example in examples:
        image_features = features(engine, image_dir, example)
        bad = example["bad"]
        scores.append(float(engine.noun_score(image_features, bad["prefix"], bad["surface"])))
    save_checkpoint(path, {"signature": signature, "scores": scores})
    return scores


def train(engine, examples, order, accumulation, image_dir, output, signature, reference=None, correction=False):
    arm = "caption_visual"
    if correction:
        arm = "negative_visual" if reference is not None else "ce_visual"
    final = output / "adapter.pt"
    if final.exists():
        existing = load_adapter(final)
        if existing["signature"] != signature:
            raise ValueError("Existing adapter was produced by different inputs. Use another output directory.")
        return existing

    parameters = [param for param in engine.model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(init_scale=128., enabled=engine.device.type == "cuda")
    checkpoint_path = output / "checkpoint.pt"
    cursor = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError("Training inputs changed. Use another output directory.")
        cursor = checkpoint["cursor"]
        if not 0 <= cursor <= len(order) or (cursor % accumulation and cursor != len(order)):
            raise ValueError("Invalid checkpoint update cursor.")
        restore(engine.adapters, checkpoint["adapters"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    for begin in range(cursor, len(order), accumulation):
        indices = order[begin:begin + accumulation]
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(indices, start=begin):
            example = examples[index]
            if correction:
                image_features = features(engine, image_dir, example)
                captions = example["captions"]
                ce = engine.caption_loss(image_features, captions[position % len(captions)])
                scaler.scale(ce / len(indices)).backward()
                if reference is not None:
                    bad = example["bad"]
                    score = engine.noun_score(image_features, bad["prefix"], bad["surface"])
                    loss = NOUN_WEIGHT * negative_loss(score, reference[index])
                    scaler.scale(loss / len(indices)).backward()
            else:
                for entry in example:
                    ce = engine.caption_loss(features(engine, image_dir, entry), entry["caption"])
                    scaler.scale(ce / (len(example) * len(indices))).backward()

        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if not math.isfinite(float(norm)) or float(norm) <= 0:
            raise FloatingPointError("Missing or non-finite adapter gradient.")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale or any(not p.isfinite().all() for p in parameters):
            raise FloatingPointError("The adapter update did not finish with finite parameters.")

        cursor = begin + len(indices)
        save_checkpoint(checkpoint_path, {
            "signature": signature, "cursor": cursor, "adapters": adapter_state(engine.adapters),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        })
        print(f"{arm}: processed {cursor}/{len(order)} training entries", flush=True)

    result = {
        "scope": "visual", "arm": arm, "layers": list(LAYERS), "rank": RANK,
        "signature": signature, "updates": math.ceil(len(order) / accumulation),
        "adapters": adapter_state(engine.adapters),
    }
    save_checkpoint(final, result)
    return result


def run_training(args):
    correction = args.command == "train-correction"
    manifest = load_json(args.manifest)
    if correction:
        examples = manifest["examples"]
        if not examples or any(not row["captions"] or not row["bad"]["surface"].strip() for row in examples):
            raise ValueError("Each correction example needs a true caption and an absent-object phrase.")
        images = examples
        order, accumulation = correction_order(len(examples)), 4
        initial_sha = sha256_file(args.initial)
        recorded_sha = manifest.get("source_adapter_sha256", manifest.get("specification", {}).get("source_adapter_sha256"))
        if recorded_sha and initial_sha != recorded_sha:
            raise ValueError("The caption checkpoint differs from the one used to prepare the correction manifest.")
    else:
        examples = manifest["pairs"]
        if not examples or any(len(pair) != 2 for pair in examples):
            raise ValueError("Expected pairs of two images with captions.")
        images = [item for pair in examples for item in pair]
        order, accumulation = caption_order(len(examples)), 8
        initial_sha = None

    validate_images(images, args.images)
    signature_fields = {
        "format": 1, "manifest": manifest, "base_config": sha256_file(args.model / "config.json"),
        "base_path": str(args.model.resolve()), "initial": initial_sha,
        "caption_only": args.caption_only, "stage": args.command,
    }
    signature = hashlib.sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()

    with training_directory(args.out):
        seed = 53 if correction else 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        engine = NativeVMC(args.model)
        reference = None
        if correction:
            restore(engine.adapters, load_adapter(args.initial)["adapters"])
            if not args.caption_only:
                reference = score_reference(engine, examples, args.images, args.out, signature)
        train(engine, examples, order, accumulation, args.images, args.out, signature, reference, correction)
    print(args.out / "adapter.pt")
