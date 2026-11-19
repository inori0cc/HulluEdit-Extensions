"""Training manifests and archived COCO caption selection."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256_file(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path):
    return Path(path).read_text(encoding="utf-8")


def load_json(path):
    return json.loads(read_json(path))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def recover_caption_pairs(index_path, coco_captions):
    index = load_json(index_path)
    if sha256_file(coco_captions) != index["coco_annotations_sha256"]:
        raise ValueError("Caption annotations differ from the archived COCO val2014 file.")
    lookup = defaultdict(dict)
    for row in load_json(coco_captions)["annotations"]:
        caption = row["caption"].strip()
        lookup[row["image_id"]][sha256_text(caption)] = caption
    groups = []
    for group in index["pairs"]:
        images = []
        for entry in group:
            image_id = entry["image_id"]
            caption = lookup[image_id].get(entry["caption_sha256"])
            if caption is None:
                raise ValueError(f"Exact archived caption not found for image {image_id}.")
            images.append({"image_id": image_id, "image": entry["image"],
                           "sha256": entry["sha256"], "caption": caption})
        groups.append(images)
    return {"pairs": groups}


def extract_correction_examples(cache, initial):
    """Convert the original fact-preservation cache to the stage-two schema."""
    cache = Path(cache)
    completed = load_json(cache / "COMPLETE.json")
    if not completed.get("valid_experiment") or not completed.get("evaluated_cases"):
        raise ValueError("Expected a valid, completed fact-preservation cache.")
    if completed["source_adapter_sha256"] != sha256_file(initial):
        raise ValueError("The cached errors were generated with a different caption checkpoint.")
    examples = []
    with (cache / "baseline.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["signature"] != completed["signature"]:
                raise ValueError("Cached caption signature differs from COMPLETE.json.")
            if not row["eligible"]:
                continue
            case = row["case"]
            if case.get("image_id", row["image_id"]) != row["image_id"]:
                raise ValueError("Case and image IDs differ.")
            if case["bad"]["category"] in set(case["ground_truth"]):
                raise ValueError("The selected error names a labeled object.")
            captions = [text for text in row["human_captions"] if 1 <= len(text.split()) <= 35]
            examples.append({
                "image_id": row["image_id"], "image": row["image"], "sha256": row["sha256"],
                "captions": captions, "bad": {key: case["bad"][key] for key in ("prefix", "surface")},
            })
    return {"examples": examples, "source_adapter_sha256": completed["source_adapter_sha256"]}


def validate_images(rows, image_dir):
    seen = set()
    for row in rows:
        image_id = row.get("image_id", row["image"])
        if image_id in seen:
            raise ValueError(f"Duplicate training image {image_id}.")
        seen.add(image_id)
        path = image_dir / row["image"]
        digest = sha256_file(path)
        if row.get("sha256", digest) != digest:
            raise ValueError(f"Image bytes differ from the manifest for {path}.")
        row["sha256"] = digest
