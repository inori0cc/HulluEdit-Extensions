# Preference-Constrained Direction Editing (PDE)

PDE adds an inference-time step to HulluEdit. For each beam, it searches within the prior subspace for an update that reduces the largest centered logit change relative to the unedited model. The search bounds the size of the update and retains a fraction of HulluEdit's preference signal. The beam uses the HulluEdit logits if the PDE candidate fails the numerical checks. Model weights stay fixed.

## Setup

Use Linux, Python 3.10, an NVIDIA GPU with a CUDA 12.4-compatible driver, and Java for BLEU tokenization. The reference hardware is a 24 GB RTX 4090.

```bash
git clone https://github.com/inori0cc/HulluEdit-Extensions.git
cd HulluEdit-Extensions
bash PDE/scripts/setup.sh
```

Setup creates `PDE/.venv`, installs the pinned dependencies, and applies the required HulluEdit compatibility files to Transformers 4.31.0 in that environment.

## Data and checkpoints

Download the [COCO 2014 validation images and train/val annotations](https://cocodataset.org/#download) and extract them under the same root.

```text
/path/to/coco/
├── annotations/
│   ├── captions_val2014.json
│   └── instances_val2014.json
└── val2014/
    └── COCO_val2014_*.jpg
```

The runner downloads the pinned [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/4481d270cc22fd5c4d1bb5df129622006ccd9234) and [CLIP ViT-L/14 336px](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/ce19dc912ca5cd21c8a653c79e251e808ccabcd1) checkpoints to your Hugging Face cache. To use local snapshots, pass `--model-path` and `--clip-path`.

## Run CHAIR evaluation

```bash
bash PDE/scripts/run_chair.sh \
  --coco-root /path/to/coco \
  --output-dir outputs/pde-chair
```

Add `--limit 1 --generate-only` and use a fresh output directory for a single-image inference check. The full run writes `captions.jsonl` and `metrics.json`. The metrics are percentages. BLEU-1 is computed for each caption using the COCO tokenizer and averaged over images.

To score an existing caption file, use the scoring entry point.

```bash
PDE/.venv/bin/python PDE/score_chair.py \
  --captions outputs/pde-chair/captions.jsonl \
  --annotations /path/to/coco/annotations \
  --output outputs/pde-chair/metrics-rerun.json
```

The [CHAIR configuration](configs/chair_llava15_7b.yaml) uses 500 ordered COCO validation images selected with seed 0, FP16 inference, 3 beams, and at most 64 new tokens. PDE extracts the evidence and prior subspaces at layer 24 and edits the final hidden state. At prompt prefill, the evidence cache includes the prefix after BOS and all visual tokens. The prior cache includes the final 34 visual tokens and the post-image prompt before the final prompt position. PDE uses rank 5, a radius factor of 1.25, preference retention of 0.5, and an interpolation coefficient of 0.25.

## Code

- [Beam editor](beam_pde.py) and [direction solver](direction_solver.py)
- [LLaVA inference integration](hulluedit/engines/llava7b.py)
- [Generation runner](run_chair.py) and [CHAIR scoring](score_chair.py)
- [Upstream acknowledgments and license notices](licenses/README.md)
