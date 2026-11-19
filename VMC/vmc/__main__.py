"""Command-line entry points for VMC."""
import argparse
from pathlib import Path

from .data import extract_correction_examples, recover_caption_pairs, write_json


def main():
    parser = argparse.ArgumentParser(description="Visual Memory Calibration for native LLaVA-1.5-7B.")
    commands = parser.add_subparsers(dest="command", required=True)

    captions = commands.add_parser("prepare-captions", help="recover first-stage caption pairs from a supplied index")
    captions.add_argument("--index", type=Path, required=True, help="path to a local caption-pair index")
    captions.add_argument("--coco-captions", type=Path, required=True)
    captions.add_argument("--out", type=Path, required=True)

    correction = commands.add_parser("prepare-correction", help="convert a local correction cache to a training manifest")
    correction.add_argument("--cache", type=Path, required=True)
    correction.add_argument("--initial", type=Path, required=True)
    correction.add_argument("--out", type=Path, required=True)

    for name in ("train-caption", "train-correction"):
        training = commands.add_parser(name)
        training.add_argument("--model", type=Path, required=True)
        training.add_argument("--images", type=Path, required=True)
        training.add_argument("--manifest", type=Path, required=True)
        training.add_argument("--out", type=Path, required=True)
        training.set_defaults(caption_only=False)
        if name == "train-correction":
            training.add_argument("--initial", type=Path, required=True)
            training.add_argument("--caption-only", action="store_true", help="omit the absent-object penalty")

    inference = commands.add_parser("infer", help="generate from one image")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--adapter", type=Path, required=True)
    inference.add_argument("--image", type=Path, required=True)
    inference.add_argument("--prompt", default="Describe this image in detail.")
    inference.add_argument("--reference", type=Path, help="optional reference image for contrastive decoding")
    inference.add_argument("--max-tokens", type=int, default=128)

    for name in ("pope", "generate-captions"):
        benchmark = commands.add_parser(name)
        benchmark.add_argument("--model", type=Path, required=True)
        benchmark.add_argument("--adapter", type=Path, help="omit for the unmodified base model")
        benchmark.add_argument("--images", type=Path, required=True)
        benchmark.add_argument("--manifest", type=Path, required=True)
        benchmark.add_argument("--out", type=Path, required=True)
        references = benchmark.add_mutually_exclusive_group()
        references.add_argument("--reference-bank", type=Path, help="ordered image list for contrastive generation")
        if name == "pope":
            references.add_argument("--text-reference", action="store_true", help="use an image-free comparison prompt")
            benchmark.add_argument("--metrics", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare-captions":
        write_json(args.out, recover_caption_pairs(args.index, args.coco_captions))
    elif args.command == "prepare-correction":
        write_json(args.out, extract_correction_examples(args.cache, args.initial))
    elif args.command in {"train-caption", "train-correction"}:
        from .train import run_training
        run_training(args)
    else:
        from .evaluate import infer, run_benchmark
        if args.command == "infer":
            infer(args)
        else:
            run_benchmark(args)


if __name__ == "__main__":
    main()
