"""Score generated COCO captions with Nullu CHAIR and per-image BLEU-1."""

import argparse
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory


def score_chair(captions_path, annotations_dir, output_path):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from nullu_chair import CHAIR

    if output_path.exists():
        raise FileExistsError(f"Score file already exists: {output_path}")

    with captions_path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    image_ids = {row["image_id"] for row in rows}
    if not rows or len(image_ids) != len(rows):
        raise ValueError("Expected one caption for each unique COCO image ID.")

    captions = json.loads((annotations_dir / "captions_val2014.json").read_text())
    instances = json.loads((annotations_dir / "instances_val2014.json").read_text())
    references = {row["image_id"]: [] for row in rows}
    for annotation in captions["annotations"]:
        image_id = annotation["image_id"]
        if image_id in references:
            references[image_id].append({"caption": annotation["caption"]})
    if any(not captions for captions in references.values()):
        raise ValueError("Missing reference captions for one or more COCO image IDs.")
    hypotheses = {row["image_id"]: [{"caption": row["caption"]}] for row in rows}

    tokenizer = PTBTokenizer()
    corpus_bleu, per_image = Bleu(4).compute_score(
        tokenizer.tokenize(references), tokenizer.tokenize(hypotheses)
    )
    formatted = {
        "overall": {f"Bleu_{j + 1}": float(score) for j, score in enumerate(corpus_bleu)},
        "imgToEval": {
            str(row["image_id"]): {
                **row,
                **{f"Bleu_{j + 1}": float(scores[i]) for j, scores in enumerate(per_image)},
            }
            for i, row in enumerate(rows)
        },
    }
    evaluator = CHAIR(image_ids, captions, instances)
    evaluator.get_annotations()
    with TemporaryDirectory() as temp:
        scores_input = Path(temp) / "chair_input.json"
        scores_input.write_text(json.dumps(formatted))
        scores = evaluator.compute_chair(str(scores_input))["overall_metrics"]

    metrics = {
        "num_images": len(rows),
        "CHAIR_s": 100 * scores["CHAIRs"],
        "CHAIR_i": 100 * scores["CHAIRi"] if scores["CHAIRi"] is not None else None,
        "BLEU_1_per_image": 100 * mean(per_image[0]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as output:
        json.dump(metrics, output, indent=2)
        output.write("\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score_chair(args.captions, args.annotations, args.output)


if __name__ == "__main__":
    main()
