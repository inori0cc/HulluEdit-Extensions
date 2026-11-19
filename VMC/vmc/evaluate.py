"""Generate captions or evaluate POPE with the report's Yes/No parser."""
import json
from collections import Counter, defaultdict

from PIL import Image

from .data import load_json, sha256_file, sha256_text, write_json
from .model import CAPTION_QUESTION, YES_NO, NativeVMC, load_adapter, restore


def open_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def make_engine(model, adapter):
    engine = NativeVMC(model)
    if adapter is not None:
        restore(engine.adapters, load_adapter(adapter)["adapters"])
    return engine


def select_reference(bank, image_id):
    identity = int(image_id) if str(image_id).isdigit() else str(image_id)
    key = json.dumps(["swap", identity], sort_keys=True, ensure_ascii=False)
    index = int(sha256_text(key)[:12], 16) % len(bank)
    if str(bank[index]["image_id"]) == str(image_id):
        index = (index + 1) % len(bank)
    return bank[index]


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def infer(args):
    engine = make_engine(args.model, args.adapter)
    reference = open_image(args.reference) if args.reference else None
    result = engine.generate(open_image(args.image), args.prompt, args.max_tokens, reference)
    print(json.dumps(result, ensure_ascii=False))


def score_answers(counts):
    tp = counts["yes", "yes"]
    tn = counts["no", "no"]
    fp = counts["no", "yes"]
    fn = counts["yes", "no"]
    invalid_pos = counts["yes", "invalid"]
    invalid_neg = counts["no", "invalid"]
    n = sum(counts.values())

    def percent(numerator, denominator):
        return 100 * numerator / max(denominator, 1)

    return {
        "n": n,
        "accuracy": percent(tp + tn, n),
        "precision": percent(tp, tp + fp + invalid_neg),
        "recall": percent(tp, tp + fn + invalid_pos),
        "f1": percent(2 * tp, 2 * tp + fp + fn + invalid_pos + invalid_neg),
        "invalid_pct": percent(invalid_pos + invalid_neg, n),
        "tp": tp, "tn": tn, "fp_valid": fp, "fn_valid": fn,
        "invalid_positive": invalid_pos, "invalid_negative": invalid_neg,
    }


def run_benchmark(args):
    pope = args.command == "pope"
    rows = list(records(args.manifest))
    if not rows:
        raise ValueError("The benchmark manifest is empty.")
    bank = load_json(args.reference_bank) if args.reference_bank else None
    if isinstance(bank, dict):
        bank = bank["items"]
    if bank is not None and (len(bank) < 3 or len({str(row["image_id"]) for row in bank}) != len(bank)):
        raise ValueError("Reference bank needs at least three distinct image IDs.")
    if pope and any(row["label"].lower() not in {"yes", "no"} for row in rows):
        raise ValueError("POPE labels must be Yes or No.")
    if pope and not args.metrics:
        raise ValueError("Pass --metrics for the POPE summary.")
    if args.out.exists() or (pope and args.metrics.exists()):
        raise FileExistsError("Choose fresh output paths for evaluation.")

    image_hashes = {row['image']: row['sha256'] for row in rows + (bank or []) if 'sha256' in row}
    for name, expected_hash in image_hashes.items():
        if sha256_file(args.images / name) != expected_hash:
            raise ValueError(f'Benchmark image differs from the manifest: {name}.')

    engine = make_engine(args.model, args.adapter)
    counts = defaultdict(Counter)
    cache = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        for row in rows:
            image_id = row["image_id"]
            question = row["question"] if pope else CAPTION_QUESTION
            foil = select_reference(bank, image_id) if bank is not None else None
            if pope:
                request = (row["image"], question, foil["image_id"] if foil else None)
                if request not in cache:
                    reference = open_image(args.images / foil["image"]) if foil else None
                    cache[request] = engine.generate(
                        open_image(args.images / row["image"]), question, 64, reference,
                        adapted=args.adapter is not None, early_answer=True, text_reference=args.text_reference,
                    )["text"]
                caption = cache[request]
                match = YES_NO.match(caption)
                prediction = match.group(1).lower() if match else "invalid"
                label = row["label"].lower()
                counts[row["split"]][label, prediction] += 1
                result = {"split": row["split"], "question": question, "label": label, "prediction": prediction}
            else:
                reference = open_image(args.images / foil["image"]) if foil else None
                caption = engine.generate(
                    open_image(args.images / row["image"]), question, 128, reference,
                    adapted=args.adapter is not None,
                )["text"]
                result = {}
            result.update(image_id=image_id, caption=caption, reference_id=foil["image_id"] if foil else None)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    if pope:
        write_json(args.metrics, {split: score_answers(values) for split, values in sorted(counts.items())})
