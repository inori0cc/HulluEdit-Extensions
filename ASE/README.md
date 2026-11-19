# Audio Subspace Editing (ASE)

ASE adapts HulluEdit to audio question answering on Qwen2-Audio and Qwen2.5-Omni. It uses audio-token states to estimate the evidence subspace and earlier text states to estimate the prior subspace. The edit runs before the decoder's final normalization. Qwen2.5-Omni also supports PDE at the output head.

## Setup

Use Linux, Python 3.11 or newer, and an NVIDIA CUDA GPU with enough memory for the 7B checkpoint. From the repository root, create a separate environment.

```bash
cd ASE
python3.11 -m venv .venv
source .venv/bin/activate
```

Install a CUDA build of PyTorch 2.6 or newer and its matching torchvision build using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install the generation dependencies.

```bash
python -m pip install -r requirements.txt
python generation/run_generation.py --help
```

## Run audio question answering

```bash
python generation/run_generation.py \
  --model qwen2-audio --audio /path/to/clip.wav \
  --question 'What sounds do you hear?' \
  --out outputs/audio_predictions.jsonl
```

The default generates both the baseline and audio HulluEdit answers. To compare audio HulluEdit and PDE on a batch of questions with Omni, run the following command.

```bash
python generation/run_generation.py \
  --model qwen2.5-omni --tasks inputs/tasks.jsonl \
  --method hulluedit_audio --method pde_audio --alpha 0.25 \
  --out outputs/omni_predictions.jsonl
```

Each task is one JSON object per line. For local audio, `audio` is relative to the task file. For [MISHAP-Bench](https://huggingface.co/datasets/hallulalm/MISHAP-Bench) audio, supply the file's repository path as `audio_path`.

```json
{"id":"example","audio":"clips/example.wav","question":"What sounds do you hear?"}
```

The runner downloads pinned [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/tree/0a095220c30b7b31434169c3086508ef3ea5bf0a) or [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/ae9e1690543ffd5c0221dc27f79834d0294cba00) checkpoints. Pass `--model-path` for a local snapshot. For gated dataset access, use the standard Hugging Face login or `HF_TOKEN`.

Audio is resampled to mono at 16 kHz and clipped to 30 seconds. The default decoding is greedy with seed 0 and a 512-token budget. Audio HulluEdit uses anchor layer 26, evidence/prior ranks 8/5, and a 512-token text window. PDE uses radius factor 1.25 and preference retention 0.5. Set `--alpha 1` for the full PDE update or `--alpha 0.25` for the reduced update.

## Optional judging

The judge accepts the generated predictions, task metadata, and a running OpenAI-compatible vLLM endpoint. Serve a model that supports structured output and tool calling. The reference judge was [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/tree/f0b7c9e722f5565102fff8481c99e4d86ae099c7).

```bash
python -m pip install -r judge/requirements.txt
python judge/run_judge.py \
  --predictions outputs/omni_predictions.jsonl --metadata inputs/metadata.jsonl \
  --base-url http://localhost:8000/v1 --out outputs/judgments.jsonl
```

Each metadata row supplies `id`, `question`, `domain`, `hallucination_type` (`context` or `knowledge`), `audio_description`, and `grounded_behavior`. Use `--examples` to supply a local JSON object of worked examples grouped by hallucination type. Use `--web-search` to enable retrieval for knowledge tasks with `strategy: "niche_specifics"`.

The generation entry point uses the corrected audio editor for all inputs. Recreating the historical cohorts requires the original task manifests, and reproducing the historical judge setup requires its worked examples and retrieval settings.

## References

- [HulluEdit](https://arxiv.org/abs/2602.22727) for evidence-consistent subspace editing
- [MISHAP-Bench](https://arxiv.org/abs/2609.33893) for audio question answering and groundedness judging. The model prompt adapters and judge helpers are adapted from that benchmark.
