# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is an MSE Deep Learning mini-challenge (FHNW/OST course "DL-MPW") on image captioning over the Flickr8k dataset. The full assignment spec, grading rubric, and submission guidelines are in `DL-MPW-ImgCaptioning.pdf` — treat it as the source of truth for what must be delivered.

All work happens in `captioning_starter/captioning_starter.ipynb`. The notebook scaffolds two required models with `TODO` stubs:

1. **Model 1 — Show and Tell** (`ShowAndTellEncoder` + `ShowAndTellCaptioner`) — CNN encoder feeds a single feature vector into an LSTM decoder. See `ShowAndTell_Paper.pdf` (Vinyals et al., 2014).
2. **Model 2 — Show, Attend and Tell** (`ShowAttendTellEncoder` + `AdditiveAttention` + `ShowAttendTellCaptioner`) — CNN encoder produces spatial feature maps; decoder attends over them at each timestep. See `ShowAttendAndTell_Paper.pdf` (Xu et al., 2015).

After both models, two "Additional Task" sections ask for self-formulated hypothesis-driven experiments.

Both models must be evaluated on the **same fixed test subset with the same BLEU setup** so results are directly comparable. Reflections (markdown cells flagged in red) must be written by the student, not generated — per the notebook's own guidelines, generative models may assist with code but not with result interpretation.

## Assignment requirements (summary of `DL-MPW-ImgCaptioning.pdf` — PDF remains source of truth)

**Deadline**: Wednesday, 3 June 2026. Group size up to 3; group members' names must appear at the top of both the notebook and the report.

**What may and may not be reused**: pretrained CNN encoders from torchvision are encouraged; small utility/training-infrastructure code is acceptable. Fully prebuilt image captioning systems or external implementations of Show-and-Tell / Show-Attend-and-Tell are **not** permitted — the core architectures must be implemented from scratch in PyTorch.

**Deliverables** (one submission per group):
- `captioning_<groupname>.ipynb` — all cells must execute without errors, outputs included, no stray intermediate cells.
- `captioning_<groupname>.pdf` — **max 4 pages**, covering approach + preprocessing, architecture/implementation choices, experimental setup, quantitative results, qualitative analysis, limitations.

**Required training setup** (apply to both models, identically where possible):
- Data augmentation on training images (small dataset → generalization matters).
- **Same** pretrained encoder backbone for both architectures; the attention model must preserve spatial feature maps (e.g. ResNet18 → 512×7×7).
- Teacher forcing during training; separate `forward()` (training) and `generate()` (inference).
- **Greedy decoding** is the required baseline for `generate()`; beam search is one of the optional extensions.
- Loss must mask `<pad>` (and handle `<eos>`) properly.
- Apply regularization: dropout, weight decay, early stopping, augmentation as appropriate.
- Track experiments with W&B (or equivalent).

**Required evaluation**:
- Quantitative: **perplexity** and **corpus-level BLEU-1 through BLEU-4** for both models on the same test subset.
- Training/validation loss curves over epochs (BLEU curves optional but encouraged).
- Qualitative analysis with both successes and failure cases; explicitly discuss missing objects, hallucinated objects, repetition, incorrect relationships, and overly generic captions.

**Additional tasks — pick at least two**:
- Beam search vs. greedy decoding (Easy)
- Attention map visualization over the image (Easy)
- Pretrained word embeddings (GloVe / Word2Vec) (Moderate)
- Bahdanau vs. dot-product attention (Moderate)
- Transformer decoder with cross-attention (Hard, bonus)

**Grading philosophy**: correct experimentation methodology and analysis weigh more than raw metric scores — limited performance is expected on a dataset this small.

## Environment

The workflow is: **edit locally in VS Code, run training on Google Colab, track experiments in Weights & Biases.** The assignment PDF only requires a runnable `.ipynb` deliverable — it does not mandate a specific runner (JupyterLab is just what `captioning_starter/requirements.txt` happens to pin).

**Local (VS Code)** — for editing, light debugging, dataloader sanity checks:

```bash
cd captioning_starter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # torch>=2.6, torchvision>=0.21, nltk, ipykernel, etc.
```

Select the `.venv` interpreter in VS Code so the notebook picks it up. CPU is fine for code paths; heavy training happens on Colab.

**Colab (training)** — open `captioning_starter.ipynb` via *File → Upload notebook* or from GitHub. Choose a GPU runtime (T4 is enough for ResNet18 + LSTM at 224×224). Colab's preinstalled torch may be older than the starter's requirements; if so, run a top-of-notebook cell:

```python
!pip install -q -U "torch>=2.6" "torchvision>=0.21" nltk wandb
```

**Dataset on Colab — copy to local SSD, do not read directly from Drive.** Reading ~8k small JPGs through a mounted Drive is dominated by per-file network latency and will bottleneck training (each epoch with `caption_sampling="all"` touches ~30k samples). Instead, keep the dataset as a single zip on Drive, copy it to `/content/` once per session, unzip locally, and point `data_dir` at the local path:

```python
from google.colab import drive
import os, shutil, zipfile

drive.mount('/content/drive')

# One-time per session: copy zip → local SSD, then unzip there.
src_zip = "/content/drive/MyDrive/captioning_data/flickr8k.zip"   # adjust path
local_zip = "/content/flickr8k.zip"
if not os.path.exists("/content/data"):
    shutil.copy(src_zip, local_zip)            # single large sequential read = fast
    with zipfile.ZipFile(local_zip) as z:
        z.extractall("/content/")               # produces /content/data/images, /content/data/splits, ...

data_dir = "/content/data/"
```

Build the zip locally once so the on-Drive layout mirrors what the dataloader expects (`data/images/`, `data/splits/`, `data/captions.txt`). Checkpoints and W&B artifacts can still go to Drive (write through the mount) — those are infrequent, large, sequential writes, which Drive handles fine.

**Weights & Biases** — the PDF explicitly recommends W&B as the experiment tracker. In Colab:

```python
from google.colab import userdata
import wandb
wandb.login(key=userdata.get("WANDB_API_KEY"))   # store the key in Colab Secrets, not in the notebook
run = wandb.init(project="dl-mpw-image-captioning", config={...})
# wandb.log({"train/loss": ..., "val/loss": ..., "val/bleu4": ...}, step=epoch)
```

Log per-epoch loss (train + val), BLEU-1..4, and the hyperparameter config. Use the same `project` and a clear `name`/`group` for the two models so the comparison is one click away.

**Device selection** — the notebook auto-selects CUDA → MPS → CPU (see the "Shared Utilities" cell). On Colab this will pick CUDA; verify before kicking off long runs.

**Before submission** — re-run the final notebook end-to-end (Colab "Run all") so all cell outputs are populated, then download the `.ipynb` for the deliverable.

## Data layout

Flickr8k images are **not** in the repo — download from https://www.kaggle.com/datasets/adityajn105/flickr8k and place the JPGs in `data/images/`. The train/test split is fixed and must not be modified:

```
data/
├── captions.txt                  # full caption list (reference only)
├── images/                       # ~8k JPGs (you must download these)
└── splits/
    ├── train_captions.csv        # image,caption pairs for training
    ├── train_images.txt          # one image_id per line
    ├── test_captions.csv
    └── test_images.txt
```

The notebook currently points at `data_dir = "../../captioning_data/data/"` (outside the repo). A copy of the splits also exists inside the repo at `captioning_starter/data/` without images. Pick whichever layout matches the local machine and keep `data_dir` consistent.

## Data pipeline (`captioning_starter/dataloader.py`, `tokenizer.py`)

These are provided helpers — generally use them as-is rather than rewriting:

- **`build_tokenizer_from_split("train", data_dir, min_freq=...)`** — builds vocabulary **only from training captions** (avoids test-set leakage). Returns a `CaptionTokenizer` with `stoi`/`itos` and special tokens at fixed indices: `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`. `encode()` automatically wraps with `<bos>...<eos>` and truncates to `max_len` (replacing the last token with `<eos>` on truncation).
- **`create_caption_dataloader(split, ..., caption_sampling=...)`** — wraps `Flickr8kCaptionsDatasetBase`. Each batch is `(images, captions, lengths, image_ids, raw_captions)` where `captions` is padded to `max_len`. `lengths` is the unpadded length **including** `<bos>`/`<eos>` — use it to mask padding in the loss (e.g. `pack_padded_sequence` or `ignore_index=tokenizer.pad_idx`).
- **`caption_sampling`** modes: `"all"` (one sample per (image, caption) pair, ~5× dataset size — typical for training), `"first"` (deterministic, one caption per image), `"random"` (one random caption per image per epoch — typical for test/eval to avoid scoring all 5 references).
- **Custom tokenizer**: pass any object implementing `CaptionTokenizerProtocol` (needs `build(captions)`, `encode(caption, max_len=...)`, `pad_idx`) via the `tokenizer=` arg.

## Conventions to follow

- **Image preprocessing**: the notebook uses ImageNet normalization (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`) at 224×224. Required when reusing pretrained torchvision backbones (ResNet, VGG, etc.) as encoders. `denormalize_image()` is provided for visualization.
- **Vocabulary hygiene**: never call `build_tokenizer_from_split("test", ...)` — vocab must come from train only. The same goes for any normalization statistics computed from data.
- **Caption sampling at eval time**: BLEU should typically be computed against **all reference captions per image** (the dataset has ~5), not the single sampled one returned by `caption_sampling="random"`. Use `dataset.captions_map[image_id]` to retrieve all references.
- **Generation**: `@torch.no_grad()` `generate()` methods should start from `<bos>` and stop at `<eos>` (or `max_len`). Greedy decoding is the baseline; beam search is a common extension worth considering for the additional tasks.
