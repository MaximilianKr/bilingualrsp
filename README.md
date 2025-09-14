# Bilingual RST Parsing — DE-MIX Fork

This fork preserves the original repository (see `README_original.md`) and adds a reproducible path to train on a mixed German RS3 corpus (PCC, APA-RST, PARADISE).

Quick links:
- Original README: `README_original.md`
- German mixed-corpus guide: `README_DEMIX.md`

Highlights in this fork:
- `DE-MIX` corpus wiring in `dmrst_parser/data_manager.py` (fixed splits under `data/de_mix_file_lists/`).
- German label normalization `germanMixed_labels` in `dmrst_parser/src/corpus/relation_set.py`.
- Utility to inventory RS3 relations: `utils/collect_rs3_relations.py`.
 - Segmentation evaluation utility with JSON output: `utils/eval_segmentation.py` (see README_DEMIX.md).

Final (segmentation reference):
- Checkpoint: `saves/de_DE-MIX_default+base_linear_hints_40` (base + linear + sentence hints).
- Token-boundary F1 on a 33-doc slice: Macro 0.776, Micro 0.744 (JSON under `results/seg_eval/`).

To train on DE-MIX, follow `README_DEMIX.md`.
