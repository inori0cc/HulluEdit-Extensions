"""Generate fresh audio QA answers with a baseline, audio HulluEdit, or PDE."""

import argparse
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path


MODELS = {
    "qwen2-audio": ("Qwen/Qwen2-Audio-7B-Instruct", "0a095220c30b7b31434169c3086508ef3ea5bf0a"),
    "qwen2.5-omni": ("Qwen/Qwen2.5-Omni-7B", "ae9e1690543ffd5c0221dc27f79834d0294cba00"),
}
DATA_REVISION = "5729324b038b828c0bedb34ac0ff163733d58079"
METHODS = ("baseline", "hulluedit_audio", "pde_audio")


def read_tasks(args):
    if args.audio is not None:
        tasks = [{"id": "audio", "audio": str(args.audio.resolve()), "question": args.question}]
        root = Path.cwd()
    else:
        with args.tasks.open(encoding="utf-8") as source:
            tasks = [json.loads(line) for line in source if line.strip()]
        root = args.tasks.resolve().parent

    if not tasks or any(not isinstance(row, dict) for row in tasks):
        raise ValueError("Expected a nonempty set of JSON task records.")
    seen = set()
    for row in tasks:
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("id", "question")):
            raise ValueError("Each task needs a nonempty string ID and question.")
        if row["id"] in seen:
            raise ValueError(f"Duplicate task ID: {row['id']}")
        seen.add(row["id"])
        sources = [key for key in ("audio", "audio_path") if key in row]
        if len(sources) != 1 or not isinstance(row[sources[0]], str) or not row[sources[0]].strip():
            raise ValueError(f"Task {row['id']} needs either audio or audio_path.")
        if "audio" in row:
            local = root / row["audio"]
            if not local.is_file():
                raise FileNotFoundError(f"Missing audio: {local}")
            row["audio"] = str(local)
    return tasks


def prepare_audio(row):
    import librosa
    import numpy as np

    path = row.get("audio")
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "hallulalm/MISHAP-Bench", row["audio_path"], repo_type="dataset", revision=DATA_REVISION
        )
    wave, rate = librosa.load(io.BytesIO(Path(path).read_bytes()), sr=16000, mono=True)
    if rate != 16000 or not wave.size or not np.isfinite(wave).all():
        raise ValueError(f"Invalid waveform for task {row['id']}")
    wave = np.ascontiguousarray(wave[:480000], dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, wave, allow_pickle=False)
    return wave, sha256(buffer.getvalue()).hexdigest()


def run(args, tasks, methods):
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Audio generation requires an NVIDIA CUDA GPU.")
    from component_editor import BASE, Intervention, numerical_checks

    if args.model == "qwen2-audio":
        from models.qwen2_audio import Qwen2Audio as Adapter
    else:
        from models.qwen2_5_omni import Qwen2_5Omni as Adapter

    model_id, revision = MODELS[args.model]
    if args.model_path is not None:
        model_id, revision = str(args.model_path), None
    wrapper = Adapter(model_id, revision)
    model = wrapper.model
    decoder = model.language_model.model if args.model == "qwen2-audio" else model.model
    config = replace(BASE, placement="pre_norm", component="full")
    if methods != ["baseline"]:
        numerical_checks()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("x", encoding="utf-8") as output, torch.inference_mode():
        for row in tasks:
            wave, digest = prepare_audio(row)
            for method in methods:
                np.random.seed(0)
                torch.manual_seed(0)
                inputs = wrapper.build_inputs([wave], [row["question"]])
                hook = None
                if method != "baseline":
                    mask = inputs.input_ids[0] == model.config.audio_token_index
                    if not mask.any():
                        raise ValueError(f"Audio tokens missing for task {row['id']}.")
                    if method == "pde_audio":
                        from pde_hook import PDEIntervention

                        hook = PDEIntervention(model, mask, config, args.alpha)
                    else:
                        hook = Intervention(decoder, mask, config)

                kwargs = wrapper.extra_gen_kwargs()
                if args.model == "qwen2-audio":
                    kwargs["repetition_penalty"] = 1.1
                try:
                    sequences = model.generate(
                        **inputs, **kwargs, do_sample=False, use_cache=True, max_new_tokens=args.max_new_tokens
                    )
                    tokens = sequences[0, wrapper.prompt_len(inputs):]
                    if hook is not None and len(hook.trace) != len(tokens):
                        raise RuntimeError("The edit hook did not observe one state per generated token.")
                    record = {
                        "id": row["id"], "variant": method, "question": row["question"],
                        "prediction": wrapper.decode(sequences, inputs)[0],
                        "array_sha256": digest, "hit_token_cap": len(tokens) == args.max_new_tokens,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"Generated {row['id']} ({method})", flush=True)
                finally:
                    if hook is not None:
                        hook.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-path", type=Path, help="Use a local checkpoint instead of the pinned Hub model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="Local audio for a single question")
    source.add_argument("--tasks", type=Path, help="JSONL manifest for batch generation")
    parser.add_argument("--question", help="Question to ask with --audio")
    parser.add_argument("--method", choices=METHODS, action="append", help="Repeat for multiple methods")
    parser.add_argument("--alpha", type=float, default=0.25, help="PDE interpolation coefficient")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True, help="New predictions JSONL path")
    args = parser.parse_args()
    methods = args.method or ["baseline", "hulluedit_audio"]
    if (args.audio is not None) != (args.question is not None):
        parser.error("--audio and --question must be used together.")
    if len(set(methods)) != len(methods):
        parser.error("Specify each method once.")
    if "pde_audio" in methods and args.model != "qwen2.5-omni":
        parser.error("PDE is available on Qwen2.5-Omni.")
    if not 0 < args.alpha <= 1 or args.max_new_tokens < 1:
        parser.error("Use 0 < --alpha <= 1 and a positive --max-new-tokens.")
    if args.out.exists():
        parser.error("--out already exists. Choose a new predictions path.")
    if args.model_path is not None and not args.model_path.is_dir():
        parser.error("--model-path must be a local checkpoint directory.")
    run(args, read_tasks(args), methods)


if __name__ == "__main__":
    main()
