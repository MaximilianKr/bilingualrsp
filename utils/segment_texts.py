import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import torch
from tqdm import tqdm
try:
    from transformers.utils import logging as hf_logging
except Exception:
    hf_logging = None

# Ensure repository root is on sys.path when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dmrst_parser.predictor import Predictor
import json as _json
try:
    import spacy as _spacy
except Exception:
    _spacy = None


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


def _compute_sent_breaks(raw_text: str, subword_offsets):
    """Compute sentence break flags aligned to subword offsets using spaCy (German).

    Returns a list of length len(subword_offsets) with 0/1 flags marking sentence ends.
    """
    if _spacy is None:
        return None
    try:
        nlp = _spacy.load("de_core_news_lg")
    except Exception:
        return None

    doc = nlp(raw_text)
    sent_end_chars = [s.end_char - 1 for s in doc.sents]
    if not sent_end_chars:
        return None

    flags = [0] * len(subword_offsets)
    si = 0
    for i, (start, end) in enumerate(subword_offsets):
        if si >= len(sent_end_chars):
            break
        # mark break when token end crosses sentence end
        if end - 1 >= sent_end_chars[si]:
            flags[i] = 1
            si += 1
    # Ensure final token is a break
    if flags:
        flags[-1] = 1
    return flags


def segment_file(model_dir: str, infile: str, outdir: str, cuda_device: int = 0, dump_breaks: bool = False, quiet: bool = True):
    pred = Predictor(model_dir, cuda_device=cuda_device)
    # Suppress common harmless warnings if quiet
    if quiet:
        try:
            pred.tokenizer.model_max_length = int(1e9)
        except Exception:
            pass
    with open(infile, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Tokenize to subwords with character offsets on raw text
    toks = pred.tokenizer(raw_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = toks['input_ids']
    sub_offs = toks['offset_mapping']
    # Optionally compute sentence breaks if the model was trained with them and lang is 'de'
    def _is_true(v):
        if isinstance(v, bool):
            return v
        try:
            return str(v).lower() in ("1", "true", "yes")
        except Exception:
            return False

    use_sent_hints = False
    try:
        cfg = _json.load(open(os.path.join(model_dir, 'config.json')))
        use_sent_hints = _is_true(cfg['model']['segmenter'].get('use_sent_boundaries')) and str(cfg['data'].get('lang')).lower() == 'de'
    except Exception:
        pass

    sent_breaks = None
    if use_sent_hints:
        flags = _compute_sent_breaks(raw_text, sub_offs)
        if flags is not None:
            sent_breaks = [flags]

    # Predict segmentation
    with torch.no_grad():
        _, _, _, _, pred_edu_breaks = pred.model.testing_loss(
            [input_ids], sent_breaks, None, None, None, None, None,
            generate_tree=False, use_pred_segmentation=True)

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

    # Write _edus.txt next to outfile
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outpath = Path(outdir) / (Path(infile).stem + '_edus.txt')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(edus) + ('\n' if edus else ''))
    if dump_breaks:
        dbg = {
            'num_subwords': len(input_ids),
            'subword_breaks': sub_breaks,
            'char_breaks': char_breaks,
            'used_sentence_hints': bool(sent_breaks is not None)
        }
        with open(str(outpath) + '.breaks.json', 'w', encoding='utf-8') as jf:
            _json.dump(dbg, jf, ensure_ascii=False, indent=2)

    return str(outpath)


def main():
    ap = argparse.ArgumentParser(description='Segment plain texts into EDUs using a trained model.')
    ap.add_argument('--model_dir', required=True, help='Path to trained run dir (contains config.json, best_weights.pt).')
    ap.add_argument('--input_dir', help='Directory containing input .txt files')
    ap.add_argument('--output_dir', help='Output directory for _edus.txt files')
    # Backward-compatible aliases
    ap.add_argument('--inputs', help='[Deprecated] Glob pattern for input .txt files (e.g., data/*.txt)')
    ap.add_argument('--out_dir', help='[Deprecated] Output directory for .edus.txt files')
    ap.add_argument('--cuda_device', type=int, default=0)
    ap.add_argument('--dump_breaks', action='store_true', help='Also write <file>.edus.txt.breaks.json with debug info')
    ap.add_argument('--verbose', action='store_true', help='Print per-file outputs and show warnings')
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

    # Configure verbosity
    if not args.verbose:
        if hf_logging is not None:
            try:
                hf_logging.set_verbosity_error()
            except Exception:
                pass
        warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")

    for fp in tqdm(files, desc='Segmenting', unit='file'):
        outp = segment_file(args.model_dir, fp, out_dir, cuda_device=args.cuda_device, dump_breaks=args.dump_breaks, quiet=not args.verbose)
        if args.verbose:
            tqdm.write(f'Wrote: {outp}')


if __name__ == '__main__':
    main()
