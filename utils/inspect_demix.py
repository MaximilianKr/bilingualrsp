"""Quick inspection for DE-MIX (or any supported corpus).

Usage:
    python utils/inspect_demix.py --corpus DE-MIX --pickle data/data_manager_demix.pickle --top 15
"""

import collections
import fire
from pathlib import Path
import sys

# Make repo root importable when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dmrst_parser.data_manager import DataManager


def main(corpus='DE-MIX', pickle='data/data_manager_demix.pickle', top=10):
    dm = DataManager(corpus=corpus).from_pickle(pickle)
    tr, dv, te = dm.get_data()
    tbl = dm.relation_table

    print(f'splits: train={len(tr.input_sentences)}, dev={len(dv.input_sentences)}, test={len(te.input_sentences)}')

    def uniq(d):
        return sorted({tbl[i] for doc in d.relation_label for i in doc})

    print(f'unique labels: train={len(uniq(tr))}, dev={len(uniq(dv))}, test={len(uniq(te))}')

    flat = []
    for doc in tr.relation_label:
        flat.extend(doc)
    c = collections.Counter(flat)
    print(f'top {top} train labels:')
    for i, n in c.most_common(top):
        print(f'  {tbl[i]}: {n}')

    print('sample golden tree (train[0]):')
    print(tr.golden_metric[0][:300])


if __name__ == '__main__':
    fire.Fire(main)
