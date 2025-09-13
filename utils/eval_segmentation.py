import argparse
import glob
import os
import re
from pathlib import Path


def load_boundaries_from_edus_file(path: str):
    """Reads an .edus or .edus.txt file and returns cumulative right-boundary indices over whitespace tokens.

    Returns:
        total_tokens (int), boundaries (set[int]) excluding the final boundary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f]

    cum = 0
    boundaries = []
    for ln in lines:
        # Tokenize on whitespace; punctuation remains attached which matches our segmentation outputs
        n = len(ln.split()) if ln else 0
        if n == 0:
            continue
        cum += n
        boundaries.append(cum - 1)

    # Remove final boundary (end of document) for standard segmentation metrics
    if boundaries:
        boundaries = boundaries[:-1]
    return cum, set(boundaries)


def f1_from_sets(pred: set, gold: set):
    tp = len(pred & gold)
    p = len(pred)
    g = len(gold)
    prec = tp / p if p > 0 else 0.0
    rec = tp / g if g > 0 else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1, tp, p, g


def main():
    ap = argparse.ArgumentParser(description='Evaluate segmentation F1 by comparing predicted EDUs to gold EDUs.')
    ap.add_argument('--pred_dir', required=True, help='Directory with predicted files (e.g., <name>.edus.txt or <name>_edus.txt)')
    ap.add_argument('--gold_dir', required=True, help='Directory with gold files (e.g., <name>.edus or <name>_edus)')
    args = ap.parse_args()

    def collect_map(folder):
        # Match names like: <name>.edus.txt, <name>.edus, <name>_edus.txt, <name>_edus
        rx = re.compile(r"^(?P<base>.+?)(?:[._]edus)(?:\.txt)?$", re.IGNORECASE)
        mapping = {}
        for fp in sorted(glob.glob(os.path.join(folder, '*'))):
            if not os.path.isfile(fp):
                continue
            m = rx.match(os.path.basename(fp))
            if not m:
                continue
            base = m.group('base')
            mapping.setdefault(base, fp)
        return mapping

    pred_map = collect_map(args.pred_dir)
    gold_map = collect_map(args.gold_dir)

    if not pred_map:
        raise SystemExit(f'No prediction EDU files found in {args.pred_dir}')

    macro_prec = []
    macro_rec = []
    macro_f1 = []
    tp_sum = p_sum = g_sum = 0

    missing = []
    per_file = []

    for name, pp in sorted(pred_map.items()):
        gold_path = gold_map.get(name)
        if not gold_path:
            missing.append(name)
            continue

        tot_pred, pred_b = load_boundaries_from_edus_file(pp)
        tot_gold, gold_b = load_boundaries_from_edus_file(gold_path)
        prec, rec, f1, tp, p, g = f1_from_sets(pred_b, gold_b)
        per_file.append((name, prec, rec, f1, p, g, tp))
        macro_prec.append(prec)
        macro_rec.append(rec)
        macro_f1.append(f1)
        tp_sum += tp
        p_sum += p
        g_sum += g

    if missing:
        print(f'Missing gold files for {len(missing)} predictions:')
        for n in missing:
            print(f'  - {n}')

    if per_file:
        print('Per-file (name, P, R, F1, |pred|, |gold|, TP):')
        for name, prec, rec, f1, p, g, tp in per_file:
            print(f'{name}\tP={prec:.3f}\tR={rec:.3f}\tF1={f1:.3f}\t|pred|={p}\t|gold|={g}\tTP={tp}')

        macro_p = sum(macro_prec)/len(macro_prec)
        macro_r = sum(macro_rec)/len(macro_rec)
        macro_f = sum(macro_f1)/len(macro_f1)
        micro_p = tp_sum / p_sum if p_sum > 0 else 0.0
        micro_r = tp_sum / g_sum if g_sum > 0 else 0.0
        micro_f = 0.0 if micro_p + micro_r == 0 else 2*micro_p*micro_r/(micro_p+micro_r)

        print('\nSummary:')
        print(f'Macro: P={macro_p:.3f}\tR={macro_r:.3f}\tF1={macro_f:.3f}\tN={len(per_file)}')
        print(f'Micro: P={micro_p:.3f}\tR={micro_r:.3f}\tF1={micro_f:.3f}\tTP={tp_sum}\t|pred|={p_sum}\t|gold|={g_sum}')


if __name__ == '__main__':
    main()
