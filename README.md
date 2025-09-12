# Bilingual RST Parsing — DE-MIX Fork

This fork preserves the original repository (see `README_original.md`) and adds a reproducible path to train on a mixed German RS3 corpus (PCC, APA-RST, PARADISE).

Quick links:
- Original README: `README_original.md`
- German mixed-corpus guide: `README_DEMIX.md`

Highlights in this fork:
- `DE-MIX` corpus wiring in `dmrst_parser/data_manager.py` (fixed splits under `data/de_mix_file_lists/`).
- German label normalization `germanMixed_labels` in `dmrst_parser/src/corpus/relation_set.py`.
- Utility to inventory RS3 relations: `utils/collect_rs3_relations.py`.

To train on DE-MIX, follow `README_DEMIX.md`.

