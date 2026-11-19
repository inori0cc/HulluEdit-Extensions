"""Generate LLaVA-1.5-7B captions with PDE and evaluate them on COCO CHAIR."""

import argparse
from dataclasses import fields
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil

from transformers_compat import check_transformers


CONFIG = Path(__file__).resolve().parent / "configs" / "chair_llava15_7b.yaml"
COHORT_SHA256 = "31a72cefecff2a7eaf2c0c2206239ce61a76480f48a74901fad44dc0349389ea"
EXPECTED_CACHE = {
    "visual": [35, 611],
    "u_input": [1, 611],
    "text": [611, 624],
    "expanded_prompt_length": 625,
    "prior_visual_positions": list(range(577, 611)),
}


def select_images(annotations, config, limit):
    by_id = {image["id"]: image["file_name"] for image in annotations["images"]}
    selected = random.Random(config["seed"]).sample(list(by_id), config["num_samples"])
    if sha256(json.dumps(selected).encode()).hexdigest() != COHORT_SHA256:
        raise ValueError("Unexpected image order in captions_val2014.json. Use the original COCO 2014 annotations.")
    return [(image_id, by_id[image_id]) for image_id in selected[:limit]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", required=True, type=Path, help="COCO root containing val2014/ and annotations/")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for captions and scores")
    parser.add_argument("--model-path", type=Path, help="Local LLaVA-1.5-7B directory (default: download the pinned revision)")
    parser.add_argument("--clip-path", type=Path, help="Local CLIP vision encoder directory (default: download the pinned revision)")
    parser.add_argument("--limit", type=int, help="Use the first N images of the fixed 500-image cohort")
    parser.add_argument("--generate-only", action="store_true", help="Skip CHAIR and BLEU-1 scoring")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(CONFIG.read_text())
    limit = args.limit if args.limit is not None else config["num_samples"]
    if not 1 <= limit <= config["num_samples"]:
        parser.error("--limit must be between 1 and 500")
    if args.output_dir.exists():
        parser.error("--output-dir already exists. Choose a new output directory.")
    for model_path in (args.model_path, args.clip_path):
        if model_path is not None and not model_path.is_dir():
            parser.error(f"Model directory does not exist: {model_path}")

    annotations = args.coco_root / "annotations"
    caption_annotations = annotations / "captions_val2014.json"
    if not caption_annotations.is_file():
        parser.error(f"Missing {caption_annotations}")
    if not args.generate_only:
        if not (annotations / "instances_val2014.json").is_file():
            parser.error(f"Missing {annotations / 'instances_val2014.json'}")
        if not shutil.which("java"):
            parser.error("Java is required for BLEU tokenization. Install Java or use --generate-only.")

    images = select_images(json.loads(caption_annotations.read_text()), config, limit)
    image_dir = args.coco_root / "val2014"
    for _, filename in images:
        if not (image_dir / filename).is_file():
            parser.error(f"Missing COCO image: {image_dir / filename}")

    check_transformers()
    import torch
    if not torch.cuda.is_available():
        parser.error("PDE generation requires an NVIDIA CUDA GPU.")

    from huggingface_hub import snapshot_download
    from hulluedit.engines.llava7b import EngineConfig, LLaVAHullueditEngine
    from hulluedit.steer import HullueditConfig
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    model_dir = args.model_path or snapshot_download(
        repo_id=config["model_id"],
        revision=config["model_revision"],
        allow_patterns=["*.json", "tokenizer.model", "pytorch_model*.bin"],
    )
    clip_dir = args.clip_path or snapshot_download(
        repo_id=config["clip_id"],
        revision=config["clip_revision"],
        allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    )
    os.environ["PDE_CLIP_PATH"] = str(Path(clip_dir).resolve())

    engine_config = EngineConfig(
        model_name=str(model_dir),
        anchor_layer=config["anchor_layer"],
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        num_beams=config["num_beams"],
        precision=config["precision"],
    )
    edit_config = HullueditConfig(**{
        field.name: config[field.name] for field in fields(HullueditConfig) if field.name in config
    })
    engine = LLaVAHullueditEngine(engine_config, edit_config)
    conversation = conv_templates["llava_v1"].copy()
    conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail.")
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    caption_path = args.output_dir / "captions.jsonl"
    with caption_path.open("x") as captions:
        for index, (image_id, filename) in enumerate(images, 1):
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            result = engine.generate(prompt, str(image_dir / filename))
            if result["prompt_cache"] != EXPECTED_CACHE:
                raise RuntimeError(f"Unexpected prompt cache: {result['prompt_cache']}")
            if not result["text"]:
                raise RuntimeError(f"Generated an empty caption for COCO image {image_id}.")
            captions.write(json.dumps({"image_id": image_id, "caption": result["text"]}) + "\n")
            captions.flush()
            print(f"Generated caption {index}/{limit}")

    if not args.generate_only:
        from score_chair import score_chair
        score_chair(caption_path, annotations, args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
