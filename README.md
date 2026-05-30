# Image Captioning — DL-MPW Mini-Challenge

Image captioning on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset
for the FHNW/OST MSE Deep Learning course (DL-MPW). Two encoder–decoder models are
implemented and compared on the same fixed test split:

1. **Show and Tell** (Vinyals et al., 2014) — CNN encoder produces a single image vector
   that initialises an LSTM caption decoder.
2. **Show, Attend and Tell** (Xu et al., 2015) — CNN encoder produces spatial feature
   maps; the LSTM decoder attends over them with additive attention at each timestep.

Two further "Additional Task" sections contain self-formulated hypothesis-driven
experiments on top of the two baselines.

The full assignment, grading rubric, and submission rules are in
[`DL-MPW-ImgCaptioning.pdf`](DL-MPW-ImgCaptioning.pdf) — that PDF is the source of truth.
Reference papers are checked in as [`ShowAndTell_Paper.pdf`](ShowAndTell_Paper.pdf) and
[`ShowAttendAndTell_Paper.pdf`](ShowAttendAndTell_Paper.pdf).

## Repository layout

```
image_captioning_project/
├── DL-MPW-ImgCaptioning.pdf          # assignment spec (source of truth)
├── ShowAndTell_Paper.pdf             # Vinyals et al., 2014
├── ShowAttendAndTell_Paper.pdf       # Xu et al., 2015
├── CLAUDE.md                         # contributor / agent notes
└── captioning_starter/
    ├── captioning_starter.ipynb      # main notebook — all work lives here
    ├── dataloader.py                 # Flickr8k dataset + DataLoader helpers
    ├── tokenizer.py                  # CaptionTokenizer (<pad>/<bos>/<eos>/<unk>)
    ├── data_structure.png            # data-pipeline overview
    ├── requirements.txt
    └── data/
        ├── captions.txt              # full caption list (reference)
        ├── images/                   # JPGs — not in repo, download separately
        └── splits/                   # fixed train/test split, do not modify
            ├── train_captions.csv
            ├── train_images.txt
            ├── test_captions.csv
            └── test_images.txt
```

## Setup

```bash
cd captioning_starter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab    # then open captioning_starter.ipynb
```

Requires Python 3.10+ and `torch >= 2.6`, `torchvision >= 0.21`. The notebook
auto-selects a device in the order **CUDA → MPS → CPU** — check the printed
device before launching long training runs.

## Data

Flickr8k images are not checked in. Download from Kaggle and place the JPGs in
`captioning_starter/data/images/`:

```bash
# https://www.kaggle.com/datasets/adityajn105/flickr8k
# extract Images/ contents into captioning_starter/data/images/
```

The fixed train/test split under `data/splits/` must not be modified. The
notebook's `data_dir` variable points to wherever the JPGs live — adjust it to
match your local layout.

**Vocabulary hygiene:** the tokenizer is built **only from training captions**
(`build_tokenizer_from_split("train", ...)`). Never build vocab or normalisation
statistics from the test split.

## Running the experiments

Open `captioning_starter/captioning_starter.ipynb` and execute cells top to
bottom. The notebook is organised as:

1. Shared utilities — device, seeding, visualisation helpers.
2. Data pipeline walkthrough using `dataloader.py` / `tokenizer.py`.
3. **Model 1** — Show and Tell: encoder, decoder, training loop, evaluation.
4. **Model 2** — Show, Attend and Tell: encoder, additive attention, decoder,
   training loop, evaluation.
5. **Additional Task 1 / 2** — hypothesis-driven experiments.
6. Final comparison and reflection.

Both models are evaluated on the **same fixed test subset with the same BLEU
setup** so the numbers are directly comparable. BLEU is computed against all
~5 reference captions per image (use `dataset.captions_map[image_id]`), not
against the single sampled caption returned during training.

## Conventions

- **Image preprocessing:** ImageNet normalisation (`mean=(0.485, 0.456, 0.406)`,
  `std=(0.229, 0.224, 0.225)`) at 224×224 — required when reusing pretrained
  torchvision backbones. `denormalize_image()` is provided for visualisation.
- **Batch shape:** `create_caption_dataloader` yields
  `(images, captions, lengths, image_ids, raw_captions)` where `captions` is
  padded to `max_len` and `lengths` includes `<bos>`/`<eos>`. Use `lengths` (or
  `ignore_index=tokenizer.pad_idx`) to mask padding in the loss.
- **Caption sampling:** `"all"` for training (≈5× samples), `"random"` or
  `"first"` for eval to avoid scoring the same image 5×.
- **Generation:** `@torch.no_grad()` `generate()` starts from `<bos>` and stops
  at `<eos>` (or `max_len`). Greedy decoding is the baseline; beam search is a
  reasonable extension for the additional tasks.

## Submission

Per the assignment PDF, hand in the executed notebook with all outputs and
written reflections. Reflections (the red-flagged markdown cells) must be
written by the student — generative tools may assist with code but not with
result interpretation.
