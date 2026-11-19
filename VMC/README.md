# Visual Memory Calibration (VMC)

VMC trains low-rank updates to the key and value projections of a frozen LLaVA-1.5-7B model. The updates apply to image-token positions during prompt processing, so the first generated token can attend to the updated visual features.

Training has two stages. The caption stage trains on human-written captions. The correction stage keeps the caption loss and penalizes object phrases that the caption-trained model previously hallucinated. The caption-trained checkpoint supplies the reference score for each phrase.

## Setup

Use Linux, Python 3.10, an NVIDIA GPU with CUDA 12.1 support, and a local [native LLaVA-1.5-7B checkpoint](https://huggingface.co/liuhaotian/llava-v1.5-7b) with the CLIP vision tower available. Run these commands from the repository root. The separate environment uses the native LLaVA loader included under [PDE/llava](../PDE/llava).

```bash
cd VMC
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export PYTHONPATH="../PDE${PYTHONPATH:+:$PYTHONPATH}"
python -m vmc --help
```

Run the remaining commands from `VMC/` with this environment active.

## Train the adapter

For stage one, provide a JSON manifest of image pairs with human captions. Each image filename is relative to `--images`.

```json
{"pairs":[[
  {"image_id":1,"image":"a.jpg","caption":"A dog lies beside a chair."},
  {"image_id":2,"image":"b.jpg","caption":"A cat sleeps on the couch."}
]]}
```

```bash
python -m vmc train-caption \
  --model /path/to/llava-v1.5-7b --images /path/to/val2014 \
  --manifest inputs/caption_pairs.json --out outputs/caption
```

For stage two, each example specifies one image, one or more true captions, and the hallucinated phrase with the text immediately preceding it. Use separate images for training and evaluation.

```json
{"examples":[
  {"image_id":3,"image":"c.jpg","captions":["A bicycle stands by the road."],
   "bad":{"prefix":"A bicycle stands next to a","surface":"car"}}
]}
```

```bash
python -m vmc train-correction \
  --model /path/to/llava-v1.5-7b --images /path/to/val2014 \
  --manifest inputs/correction.json --initial outputs/caption/adapter.pt \
  --out outputs/vmc
```

Rerun the same command with the same inputs to resume from the last completed update. Use a fresh output directory when changing the inputs or objective. Add `--caption-only` for the matched control that uses caption loss alone.

If you have your own caption-pair index, `prepare-captions --index` can recover its captions from COCO annotations. If you have a local correction cache and its corresponding caption checkpoint, `prepare-correction` converts them to the stage-two manifest. Each subcommand lists its arguments with `--help`.

## Inference

```bash
python -m vmc infer \
  --model /path/to/llava-v1.5-7b --adapter outputs/vmc/adapter.pt \
  --image example.jpg --prompt 'Describe this image in detail.'
```

Add `--reference reference.jpg` for contrastive decoding. Both image streams use VMC and share the generated text prefix.

## Evaluation

For POPE, supply a JSON Lines manifest with one row per question and split.

```json
{"image_id":3,"image":"c.jpg","question":"Is there a bicycle in the image?","label":"yes","split":"random"}
```

```bash
python -m vmc pope \
  --model /path/to/llava-v1.5-7b --adapter outputs/vmc/adapter.pt \
  --images /path/to/val2014 --manifest inputs/pope.jsonl \
  --out outputs/pope_predictions.jsonl --metrics outputs/pope_metrics.json
```

The parser checks the initial Yes/No word. Invalid answers count as accuracy errors. Invalid answers on positive examples count against recall, those on negative examples count against precision, and both count against F1. Omit `--adapter` to evaluate the base model. Add `--text-reference` to compare with an image-free prompt.

For CHAIR caption generation, supply a JSON Lines manifest with `image_id` and `image` on each line.

```bash
python -m vmc generate-captions \
  --model /path/to/llava-v1.5-7b --adapter outputs/vmc/adapter.pt \
  --images /path/to/val2014 --manifest inputs/chair_images.jsonl \
  --out outputs/captions.jsonl
```

For either benchmark, `--reference-bank` takes an ordered JSON array of at least three distinct objects with `image_id` and `image` fields. The target ID and list order determine the reference image. For CHAIR, pass the generated captions and the matching COCO annotations to the [OPERA CHAIR evaluator](https://github.com/shikiw/OPERA).

## References

- [LLaVA-1.5](https://arxiv.org/abs/2310.03744), [LoRA](https://arxiv.org/abs/2106.09685), [NPO](https://arxiv.org/abs/2404.05868), and [Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)
- [POPE](https://arxiv.org/abs/2305.10355) and [CHAIR](https://arxiv.org/abs/1809.02156)
