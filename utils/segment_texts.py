import argparse
import glob
import os
from pathlib import Path

import torch

from dmrst_parser.predictor import Predictor


def _get_offset_mappings(tokenizer, input_ids):
    # Mirror trainer token offset approximation for sentence/EDU mapping
    subwords_str = tokenizer.convert_ids_to_tokens(input_ids)
    start, end = 0, 0
    result = []
    for subword in subwords_str:
        if subword.startswith('▁'):
            if subword != '▁':
                start += 1
        if subword == '<P>' and start > 0:
            start += 1
            end += 1
        end += len(subword)
        result.append((start, end))
        start = end
    return result


def _word_offsets(words):
    offs = []
    cur = 0
    for w in words:
        offs.append((cur, cur + len(w)))
        cur += len(w) + 1
    return offs


def _subword_to_word_breaks(word_offsets, subword_offsets, subword_breaks):
    # Map each predicted subword break j to the largest word index whose end <= subword_offsets[j].end-1
    result = []
    for j in subword_breaks:
        sw_end = subword_offsets[j][1] - 1
        wi = 0
        for idx, (_, w_end) in enumerate(word_offsets):
            if w_end - 1 <= sw_end:
                wi = idx
            else:
                break
        result.append(wi)
    # Ensure strictly increasing indices
    filtered = []
    last = -1
    for wi in result:
        if wi > last:
            filtered.append(wi)
            last = wi
    return filtered


def segment_file(model_dir: str, infile: str, outdir: str, cuda_device: int = 0):
    pred = Predictor(model_dir, cuda_device=cuda_device)
    with open(infile, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    # Basic whitespace tokenization; keep consistent with trainer char-offset logic
    words = text.split()
    text_ws = ' '.join(words)

    # Tokenize to subwords
    toks = pred.tokenizer(text_ws, add_special_tokens=False)
    input_ids = toks['input_ids']
    sub_offs = _get_offset_mappings(pred.tokenizer, input_ids)
    # Predict segmentation
    with torch.no_grad():
        _, _, _, _, pred_edu_breaks = pred.model.testing_loss(
            [input_ids], None, None, None, None, None, None, generate_tree=False, use_pred_segmentation=True)

    # pred_edu_breaks is a list with one list of subword indices
    word_offs = _word_offsets(words)
    word_breaks = _subword_to_word_breaks(word_offs, sub_offs, pred_edu_breaks[0])

    # Reconstruct EDUs by word breaks
    edus = []
    start = 0
    for wb in word_breaks:
        edus.append(' '.join(words[start:wb + 1]))
        start = wb + 1

    # Write .edus.txt next to outfile
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outpath = Path(outdir) / (Path(infile).stem + '.edus.txt')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(edus) + ('\n' if edus else ''))

    return str(outpath)


def main():
    ap = argparse.ArgumentParser(description='Segment plain texts into EDUs using a trained model.')
    ap.add_argument('--model_dir', required=True, help='Path to trained run dir (contains config.json, best_weights.pt).')
    ap.add_argument('--inputs', required=True, help='Glob pattern for input .txt files (e.g., data/*.txt)')
    ap.add_argument('--out_dir', required=True, help='Output directory for .edus.txt files')
    ap.add_argument('--cuda_device', type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(args.inputs))
    if not files:
        raise SystemExit(f'No files matched: {args.inputs}')

    print(f'Segmenting {len(files)} files with model: {args.model_dir}')
    for fp in files:
        outp = segment_file(args.model_dir, fp, args.out_dir, cuda_device=args.cuda_device)
        print(f'Wrote: {outp}')


if __name__ == '__main__':
    main()

