# DE-MIX: Training on a Mixed German RS3 Corpus

This fork adds a reproducible path to train the bilingual RST parser on a mixed German corpus (PCC, APA-RST, PARADISE), all in RS3 format, using fixed train/dev/test splits.

## Data Layout

Place your RS3 files under:

```
data/de_mix_rs3/
├── train/*.rs3
├── dev/*.rs3
└── test/*.rs3
```

Provide the corresponding split lists (paths relative to `data/de_mix_rs3`):

```
data/de_mix_file_lists/
├── train.txt   # e.g., train/pcc__maz-00001
├── dev.txt     # e.g., dev/apa__1-21-2-18-a2
└── test.txt    # e.g., test/paradise__CRE194_Blog
```

Entries may include or omit the split prefix; the loader normalizes either form.

## Relations Inventory (optional but recommended)

Collect an inventory of relation names and types across the dataset:

```
python utils/collect_rs3_relations.py --root data/de_mix_rs3 --out data/de_mix_relations.csv
```

This writes a CSV summarizing header relations and used relnames per split and subcorpus.

## Mapping and Label Space

- German label normalization is defined in `dmrst_parser/src/corpus/relation_set.py:germanMixed_labels`, extending `germanPcc_labels` with observed APA/PARADISE variants (e.g., `sameunit` → `same-unit`, `e-elaboration` → `entity-elaboration`, suffix-stripped `evaluation-n` → `evaluation`).
- Training uses the RST-DT coarse label table with nuclearity (`RelationTableRSTDT`), via a class-collapsing dictionary (`rel2class`).

### German Label Normalization (summary)
- sameunit → same-unit
- e-elaboration → entity-elaboration
- manner-means → means
- evaluation-n / evaluation-s → evaluation
- reason-n → reason
- solutionhood-n → solutionhood
- restatement-mn → restatement
- unconditional → condition
- unstated-relation → elaboration
- attribution → attribution (explicit)
- same-unit → same-unit (explicit)

## Preparing Data (RS3 → *.lisp/*.edus → *.pkl)

Build the DataManager cache for DE-MIX (creates `data/de_mix_prepared/*.lisp`, `*.edus`, and `*.pkl`):

```
python -c "from dmrst_parser.data_manager import DataManager; dm=DataManager('DE-MIX'); dm.from_rs3(); dm.save('data/data_manager_demix.pickle')"
```

## Training

Use the experiment driver, picking your transformer and model type. You can limit epochs for quick smoke runs:

```
python dmrst_parser/multiple_runs.py \
  --corpus 'DE-MIX' \
  --lang 'en' \
  --model_type 'default' \
  --transformer_name 'xlm-roberta-large' \
  --epochs 2 \
  --cuda_device 0 \
  train
```

Quick start variants:
- Base model (faster smoke run):
```
python dmrst_parser/multiple_runs.py \
  --corpus DE-MIX --lang en --model_type default \
  --transformer_name xlm-roberta-base --epochs 2 --n_runs 1 \
  --cuda_device 0 train
```
- Large model (original default):
```
python dmrst_parser/multiple_runs.py \
  --corpus DE-MIX --lang en --model_type default \
  --transformer_name xlm-roberta-large --epochs 2 --n_runs 1 --freeze_first_n 20 --lr 5e-5 \
  --cuda_device 0 train
```

Notes:
- emb_size is auto-set for common variants: 768 for xlm-roberta-base; 1024 for xlm-roberta-large. You can still override via --emb_size.
- `--model_type '+tony'` enables the ToNy segmenter (LSTM-CRF), often a strong baseline.
- Adjust `--transformer_name`/`--emb_size` for a smaller LM (e.g., `xlm-roberta-base`) to reduce compute.
- Tip: for xlm-roberta-large, freezing first ~20 layers and lowering LR (e.g., 5e-5) often stabilizes early epochs.

Long sequences and windowing:
- You may see a tokenizer warning about sequences >512 subwords; the model applies sliding windows internally (see window_size/window_padding in configs) to handle long docs.

### German optional features
- Sentence boundaries for the segmenter (optional):
  - Enable hints with `--segmenter_use_sent_boundaries 1` when training.
  - Use with `--lang de` and install spaCy German: `python -m spacy download de_core_news_lg`.
  - Example (large, short run):
```
python dmrst_parser/multiple_runs.py \
  --corpus DE-MIX --lang de --model_type default \
  --transformer_name xlm-roberta-large --epochs 8 --n_runs 1 \
  --freeze_first_n 20 --lr 1e-4 --segmenter_use_sent_boundaries 1 \
  --cuda_device 0 train
```
  - Inference with the trained run is unchanged; the model will use sentence hints internally if it was trained with them.
- LUKE/MLUKE entity spans (optional):
  - If you switch to a LUKE model (`studio-ousia/mluke-*`), Trainer extracts entity spans with spaCy.
  - For German, set `--lang de` and install spaCy German as above. This can improve performance in some setups but adds compute.

### Mixed Precision (AMP) — optional
- Enable AMP to reduce VRAM and sometimes improve throughput:
  - Add `--use_amp 1` to the training command.
  - Example (large, short run):
```
python dmrst_parser/multiple_runs.py \
  --corpus DE-MIX --lang en --model_type default \
  --transformer_name xlm-roberta-large --epochs 8 --n_runs 1 \
  --freeze_first_n 20 --lr 1e-4 --use_amp 1 --cuda_device 0 train
```
- Accuracy impact is typically negligible; early-epoch differences can occur. If AMP appears unstable, try a slightly lower LR or disable AMP.
- For details and rationale, see `docs/training_freezing_and_amp.md`.

## Inference

Given a trained run in `saves/<run_name>/` containing `config.json` and `best_weights.pt`, load and run:

```python
from dmrst_parser.predictor import Predictor

pred = Predictor('saves/<run_name>', cuda_device=0)
# Prepare a Data batch (tokenized tokens + EDU breaks) using the same DataManager pipeline.
# Then:
# batches = pred.get_batches(data, size=1)
# for batch in batches:
#     logits = pred.model(...)
```

Segmentation from plain text (utility):
- Use `utils/segment_texts.py` to turn `.txt` into `.edus.txt` with predicted EDU boundaries.
```
python utils/segment_texts.py --model_dir saves/<run_name> \
  --input_dir data/test_input --output_dir data/test_output --cuda_device 0
```
Each input file produces `<output_dir>/<name>.edus.txt` with one EDU per line. Add `--dump_breaks` to also write a small JSON with debug info. Output segments are cut at predicted character offsets (no leading spaces).

Segmentation evaluation (token-boundary F1):
```
python utils/eval_segmentation.py --pred_dir data/test_output --gold_dir data/gold_test
```
- Matches files by their base name before the EDU suffix, supporting both `<name>.edus(.txt)` and `<name>_edus(.txt)` in either folder.
- Compares right-boundary indices over whitespace tokens (ignores the final boundary). This approximates the trainer’s segmentation metric; tokenization differences can affect scores.

## Regression Checks

- After changes, re-run RuRSTB preparation to ensure nothing breaks existing flows:
```
python -c "from dmrst_parser.data_manager import DataManager; dm=DataManager('RuRSTB'); dm.from_rs3(); print('OK')"
```

- Prepare DE-MIX subset to validate mapping and outputs.

## Provenance

- The original README.md documents GUM, RST-DT, and RuRSTB workflows. This readme adds DE-MIX specifics while preserving original behavior.
- Quick inspection (optional):
```
python utils/inspect_demix.py --corpus DE-MIX --pickle data/data_manager_demix.pickle --top 15
```
Notes:
- emb_size is auto-set for common variants: 768 for xlm-roberta-base; 1024 for xlm-roberta-large. You can still override via --emb_size.
- `--model_type '+tony'` enables the ToNy segmenter (LSTM-CRF), often a strong baseline.
- Adjust `--transformer_name`/`--emb_size` for a smaller LM (e.g., `xlm-roberta-base`) to reduce compute.

Best-variant per paper (ToNy + E‑BiLSTM):
- Use `--model_type '+tony+bilstm_edus'` to enable ToNy segmenter and BiLSTM EDU encoder.
```
python dmrst_parser/multiple_runs.py \
  --corpus DE-MIX --lang en --model_type +tony+bilstm_edus \
  --transformer_name xlm-roberta-large --epochs 8 --n_runs 1 --freeze_first_n 20 --lr 1e-4 \
  --cuda_device 0 train
```
Notes: ToNy works without sentence-boundary hints by default; you can also combine it with `--segmenter_use_sent_boundaries 1` under `--lang de` if experimenting with sentence hints.
## Troubleshooting
- ModuleNotFoundError when launching training: fixed by launching trainer as a module (handled by multiple_runs). Ensure you call multiple_runs.py, not trainer.py directly.
- RuntimeError LayerNorm shape mismatch: pass matching `--emb_size` for your LM (base=768, large=1024) or rely on auto-selection.
- Unstable training with xlm-roberta-large: use `--freeze_first_n 20` and a lower `--lr` (e.g., 5e-5).
- Unknown label during prep: re-run `utils/collect_rs3_relations.py`, extend `germanMixed_labels`, and rebuild the cache.
- Tokenizer warning about sequences >512 subwords is expected; the model applies a sliding window (window_size/window_padding) at inference/training time.
