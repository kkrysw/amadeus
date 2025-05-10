# 🎼 Amadeus: MIDI Transcription with Deep Learning

Amadeus is a lightweight CRNN-based pipeline for automatic piano transcription, converting audio into symbolic note predictions. Unlike the complex and modular ReconVAT architecture, this repo emphasizes simplicity, reproducibility, and direct interpretability. 

---

## Pipeline Overview

### Preprocessing Flow

To train the model, you must generate aligned mel-spectrograms and label tensors from the MAPS dataset. This happens in **two stages**:

1. `truePreprocess.py`  
   Converts raw MAPS `.wav` + `.mid` files into `.flac` audio and `.tsv` files containing aligned pitch and timing labels.

2. `secondPre.py`  
   Converts `.flac` and `.tsv` into:
   - `*_mel.pt`: `[T, 229]` Mel spectrograms
   - `*_label.pt`: `[T, 88]` piano roll targets
   - `*_onset.pt`: `[T, 88]` onset-only binary maps

Output is saved to `true_preprocessed_tensors/` as hundreds of `.pt` files.

---

## Model Architecture

The transcription model is defined in `CRNN/model.py` and trained via `localTrain.py`.

**Key features:**
- CNN frontend → GRU → linear sigmoid heads
- Dual loss heads: one for frame-level notes, one for onsets
- Threshold sweep across sigmoid outputs per epoch
- Epoch-wise sigmoid heatmap visualization
- `mir_eval` integration for onset-based F1 scoring

---

##  Training & Evaluation

To train the CRNN:

python localTrain.py


Make sure `/content/preprocessed_tensors` contains the `mel`, `label`, and `onset` `.pt` files.

### Training outputs:
- `best_model.pt` — Best checkpoint by validation loss
- `loss_log.csv` — Tracks training + validation losses, F1, precision/recall
- `heatmap_epochX.png` — Epoch 1 sigmoid visualization
- Console output showing both frame-level and `mir_eval` F1 scores

---

##  Folder Guide

```text
amadeus/
├── CRNN/
│   ├── model/                → CRNN model definition  
│   ├── localTrain.py         → Main training entry point  
│   ├── localTrueDataset.py   → Dataset loader for raw .pt tensors  
│   ├── truePreprocess.py     → WAV + MIDI to .flac + .tsv  
│   ├── secondPre.py          → .flac + .tsv to mel/label/onset tensors  
│   ├── visualize.py          → Summary plots across MAPS categories (ISOL, UCHO, etc.)  
│   └── checkMismatches.py    → Validation script for tensor consistency  
├── weights_local/            → Saved weights + logs  
└── reconvat/                 → Deprecated — ignore  
```



---

##  Notes on Evaluation

- **Frame-level F1** scores are highly sensitive to thresholding.
- Due to the **extreme label sparsity**, models default to high precision or high recall but rarely both.
- `mir_eval.transcription` is used to compute **note onset F1** with a 50ms tolerance.

**Threshold behavior:**
- Lower threshold → higher recall, lower precision (more false positives)
- Higher threshold → higher precision, lower recall (missed notes)

---

## Final Notes

- Ignore `reconvat/` — it was part of an abandoned model.
- For a clean Colab run, extract the tensors, then run `localTrain.py`.
- MAPS dataset sparsity is the main obstacle. Despite this, our CRNN showed promising trends and interpretable failure cases.

---

## Contributors

Developed by Alex Wu, Kevin Fu, and Krystal Wu for NYU Deep Learning Spring 2025.
