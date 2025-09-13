import argparse
import glob
import os
import sys
from pathlib import Path

import torch

# Ensure repository root is on sys.path when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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


def segment_file(model_dir: str, infile: str, outdir: str, cuda_device: int = 0, dump_breaks: bool = False):
    pred = Predictor(model_dir, cuda_device=cuda_device)
    with open(infile, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Tokenize to subwords with character offsets on raw text
    toks = pred.tokenizer(raw_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = toks['input_ids']
    sub_offs = toks['offset_mapping']
    # Predict segmentation
    with torch.no_grad():
        _, _, _, _, pred_edu_breaks = pred.model.testing_loss(
            [input_ids], None, None, None, None, None, None, generate_tree=False, use_pred_segmentation=True)

    # pred_edu_breaks is a list with one list of subword indices
    sub_breaks = pred_edu_breaks[0]
    # Map subword breaks to character end positions
    char_breaks = [max(0, sub_offs[j][1] - 1) for j in sub_breaks]

    # Reconstruct EDUs by character slices on the original text
    edus = []
    start_char = 0
    for ce in char_breaks:
        # Skip inter-token whitespace at the boundary so segments don't start with a space
        while start_char < len(raw_text) and raw_text[start_char].isspace() and start_char <= ce:
            start_char += 1
        seg = raw_text[start_char:ce + 1]
        edus.append(seg)
        start_char = ce + 1

    # Write .edus.txt next to outfile
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outpath = Path(outdir) / (Path(infile).stem + '.edus.txt')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(edus) + ('\n' if edus else ''))
    if dump_breaks:
        import json
        dbg = {
            'num_subwords': len(input_ids),
            'subword_breaks': sub_breaks,
            'char_breaks': char_breaks,
        }
        with open(str(outpath) + '.breaks.json', 'w', encoding='utf-8') as jf:
            json.dump(dbg, jf, ensure_ascii=False, indent=2)

    return str(outpath)


def main():
    ap = argparse.ArgumentParser(description='Segment plain texts into EDUs using a trained model.')
    ap.add_argument('--model_dir', required=True, help='Path to trained run dir (contains config.json, best_weights.pt).')
    ap.add_argument('--input_dir', help='Directory containing input .txt files')
    ap.add_argument('--output_dir', help='Output directory for .edus.txt files')
    # Backward-compatible aliases
    ap.add_argument('--inputs', help='[Deprecated] Glob pattern for input .txt files (e.g., data/*.txt)')
    ap.add_argument('--out_dir', help='[Deprecated] Output directory for .edus.txt files')
    ap.add_argument('--cuda_device', type=int, default=0)
    ap.add_argument('--dump_breaks', action='store_true', help='Also write <file>.edus.txt.breaks.json with debug info')
    args = ap.parse_args()

    # Normalize args
    input_arg = args.input_dir or args.inputs
    out_dir = args.output_dir or args.out_dir
    if not input_arg or not out_dir:
        raise SystemExit('Please provide --input_dir and --output_dir (or legacy --inputs/--out_dir).')

    pattern = os.path.join(input_arg, '*.txt') if os.path.isdir(input_arg) else input_arg
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'No files matched: {pattern}')

    print(f'Segmenting {len(files)} files with model: {args.model_dir}')
    for fp in files:
        outp = segment_file(args.model_dir, fp, out_dir, cuda_device=args.cuda_device, dump_breaks=args.dump_breaks)
        print(f'Wrote: {outp}')


if __name__ == '__main__':
    main()
