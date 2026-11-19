# Three Extensions of HulluEdit for Hallucination Mitigation

A course project for the Fall 2026 [Trustworthy AI Models and Systems seminar](https://github.com/lydiaYchen/26TrustAI). We study how models use evidence from an image or recording when deciding what to say. The three extensions intervene at different points. VMC calibrates visual memory while the prompt is being read. PDE constrains the change in token scores during decoding. ASE transfers subspace editing to audio question answering.

HulluEdit provides the starting point. It separates evidence-bearing directions in decoder states from directions associated with language priors, then attenuates the prior and residual components during generation. See the [HulluEdit paper](https://arxiv.org/abs/2602.22727) and [upstream implementation](https://github.com/VioAgnes/HulluEdit).

## Extensions

| Extension | Intervention | Backbone | Evaluation workflow |
| --- | --- | --- | --- |
| [PDE](PDE/) · Preference-Constrained Direction Editing | Inference-time edit within the language-prior subspace | LLaVA-1.5-7B | COCO 2014 caption generation, Nullu CHAIR, and per-image BLEU-1 |
| [VMC](VMC/) · Visual Memory Calibration | Trained, image-token-only adapters for attention keys and values | LLaVA-1.5-7B | POPE scoring and caption generation for CHAIR |
| [ASE](ASE/) · Audio Subspace Editing | Audio-conditioned decoder-state editing, with optional PDE on Omni | Qwen2-Audio-7B and Qwen2.5-Omni-7B | MISHAP-Bench-style audio QA and optional groundedness judging |

## Getting started

Clone the repository, then follow the instructions for one extension. Use Linux and an NVIDIA CUDA GPU for model runs. Each extension uses its own Python environment.

```bash
git clone https://github.com/inori0cc/HulluEdit-Extensions.git
cd HulluEdit-Extensions
```

### PDE · Constrain the decoding edit

At each beam-search step, PDE uses HulluEdit's edited state as a reference. It searches for a change in the prior subspace that limits the largest change in pairwise token log odds relative to the unedited model. The search retains a specified fraction of HulluEdit's preference shift. A candidate that fails the numerical checks falls back to the HulluEdit scores.

From the repository root, install the environment with Python 3.10 available. Use Java for BLEU tokenization and supply the COCO 2014 validation images and annotations. The runner downloads the pinned LLaVA and CLIP checkpoints or accepts paths to local snapshots. [PDE setup and evaluation guide](PDE/README.md)

```bash
bash PDE/scripts/setup.sh
bash PDE/scripts/run_chair.sh \
  --coco-root /path/to/coco \
  --output-dir outputs/pde-chair
```

The default run generates captions for the fixed, ordered 500-image COCO validation cohort with seed 0, three beams, and up to 64 new tokens. It writes `captions.jsonl` and `metrics.json`. For an inference check, add `--limit 1 --generate-only` and choose a fresh output directory.

### VMC · Calibrate image memory before the first answer token

VMC trains low-rank adapters for key and value projections at image-token positions. The calibrated entries are written during the initial prompt pass, so the last prompt state can use them when it scores the first answer token. Training starts with human captions, then keeps the caption loss while penalizing object phrases previously hallucinated by the caption-trained model.

Use Python 3.10, the native LLaVA-1.5-7B checkpoint, and the CLIP vision tower. VMC loads the native LLaVA code from this repository. The guide gives the environment setup, input formats, both training commands, and evaluation commands. [VMC setup and training guide](VMC/README.md)

Once you have an adapter, run single-image inference from `VMC/` with its environment active.

```bash
python -m vmc infer \
  --model /path/to/llava-v1.5-7b \
  --adapter outputs/vmc/adapter.pt \
  --image /path/to/image.jpg \
  --prompt 'Describe this image in detail.'
```

The `pope` command writes predictions and metrics for a question manifest. The `generate-captions` command writes captions for the external CHAIR evaluator. [POPE paper](https://arxiv.org/abs/2305.10355) · [CHAIR paper](https://arxiv.org/abs/1809.02156)

### ASE · Transfer subspace editing to audio

ASE estimates an evidence subspace from audio tokens. It removes those directions from recent text states before estimating the prior subspace. Audio HulluEdit then edits the decoder state before final normalization. On Qwen2.5-Omni, an optional PDE step constrains the additional output-head update.

Set up the ASE environment with Python 3.11 or newer. Run the following from `ASE/` to generate paired baseline and audio HulluEdit answers. [ASE setup and evaluation guide](ASE/README.md)

```bash
python generation/run_generation.py \
  --model qwen2-audio \
  --audio /path/to/clip.wav \
  --question 'What sounds do you hear?' \
  --out outputs/audio_predictions.jsonl
```

For batch runs, supply a JSONL task manifest. With `--model qwen2.5-omni`, select `--method hulluedit_audio --method pde_audio` to compare the two edits. The optional judge uses task rubrics and a compatible local endpoint to score answers about music, environmental sound, and speech. [MISHAP-Bench paper](https://arxiv.org/abs/2609.33893) · [Dataset](https://huggingface.co/datasets/hallulalm/MISHAP-Bench)

## Evaluation inputs

Download checkpoints and datasets from their original sources. Supply the training and evaluation manifests required by each workflow. To recreate the course report's runs, use the original task selections and matching adapter or judge configuration where applicable.

For a paired comparison, give each method the same inputs, prompts, decoding settings, and evaluator. The guides document the files needed for each workflow. Source acknowledgments and licenses for the shared LLaVA and HulluEdit code are listed with PDE. [Upstream notices](PDE/licenses/README.md)
