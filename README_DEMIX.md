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
