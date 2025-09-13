"""
Utility: Collect relation label inventory across RS3 files.

Scans a root directory recursively for *.rs3 files, extracts:
- Header relation definitions (name + type: mononuc/multinuc)
- Used relnames in <segment> and <group> nodes

Outputs a CSV aggregated by split (train/dev/test), subcorpus (prefix before '__' in filename),
source (header/segment/group), relation_name, relation_type (if applicable), and counts.

Usage:
    python utils/collect_rs3_relations.py --root data/de_mix_rs3 --out data/de_mix_relations.csv
"""

import csv
import os
from collections import defaultdict
from pathlib import Path
import sys

import fire

from lxml import etree


def parse_xml(rs3file):
    """Parse RS3 XML with fallback encodings."""
    for enc in ["utf8", "windows-1252"]:
        try:
            xml_parser = etree.XMLParser(encoding=enc)
            rs3_xml_tree = etree.parse(rs3file, xml_parser)
            doc_root = rs3_xml_tree.getroot()
            return doc_root, rs3_xml_tree
        except Exception:
            continue
    # Final attempt without explicit encoding
    xml_parser = etree.XMLParser()
    rs3_xml_tree = etree.parse(rs3file, xml_parser)
    doc_root = rs3_xml_tree.getroot()
    return doc_root, rs3_xml_tree


def get_relations_type(rs3_xml_tree):
    """Collect relation names and their types from RS3 header."""
    relations = {}
    # Use .// to include root in search (avoids lxml FutureWarning)
    for rel in rs3_xml_tree.iterfind('.//header/relations/rel'):
        rel_name = rel.attrib.get('name', '').replace(' ', '-')
        rel_type = rel.attrib.get('type')
        if not rel_name or not rel_type:
            continue
        if rel_name not in relations:
            relations[rel_name] = set()
        relations[rel_name].add(rel_type)
    return {k: list(v) for k, v in relations.items()}


def normalize(name: str) -> str:
    if name is None:
        return ''
    return name.strip().replace(' ', '-').lower()


def subcorpus_from_basename(basename: str) -> str:
    # Expect patterns like: `apa__...`, `pcc__...`, `paradise__...`
    if '__' in basename:
        return basename.split('__', 1)[0]
    return 'unknown'


def split_from_relpath(relpath: str) -> str:
    # Expect `train/...`, `dev/...`, or `test/...`
    parts = Path(relpath).parts
    if parts:
        first = parts[0].lower()
        if first in ('train', 'dev', 'test'):
            return first
    return 'unspecified'


def collect(root: str, out: str):
    header_counts = defaultdict(int)  # (split, subcorpus, relation_name, relation_type)
    segment_counts = defaultdict(int)  # (split, subcorpus, relation_name)
    group_counts = defaultdict(int)  # (split, subcorpus, relation_name)

    rs3_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.rs3') and not fn.startswith('.'):
                rs3_files.append(os.path.join(dirpath, fn))

    for path in rs3_files:
        relpath = os.path.relpath(path, root)
        split = split_from_relpath(relpath)
        basename = os.path.basename(path)
        subcorpus = subcorpus_from_basename(os.path.splitext(basename)[0])

        # Parse RS3 with fallback encodings
        doc_root, rs3_xml_tree = parse_xml(path)

        # Header relations: count presence per file
        relations = get_relations_type(rs3_xml_tree)  # dict: name -> [types]
        for rel_name, types in relations.items():
            for t in types:
                header_counts[(split, subcorpus, normalize(rel_name), normalize(t))] += 1

        # Used relnames in segments
        for segment in doc_root.iter('segment'):
            relname = normalize(segment.attrib.get('relname', ''))
            if relname:
                segment_counts[(split, subcorpus, relname)] += 1

        # Used relnames in groups
        for group in doc_root.iter('group'):
            relname = normalize(group.attrib.get('relname', ''))
            if relname:
                group_counts[(split, subcorpus, relname)] += 1

    # Ensure output directory exists
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'subcorpus', 'source', 'relation_name', 'relation_type', 'count'])

        for (split, subcorpus, rel, t), cnt in sorted(header_counts.items()):
            writer.writerow([split, subcorpus, 'header', rel, t, cnt])

        for (split, subcorpus, rel), cnt in sorted(segment_counts.items()):
            writer.writerow([split, subcorpus, 'segment', rel, '', cnt])

        for (split, subcorpus, rel), cnt in sorted(group_counts.items()):
            writer.writerow([split, subcorpus, 'group', rel, '', cnt])

    # Print a brief summary
    uniq_header = len({k[2] for k in header_counts})
    uniq_segment = len({k[2] for k in segment_counts})
    uniq_group = len({k[2] for k in group_counts})
    print(f'Files scanned: {len(rs3_files)}')
    print(f'Unique header relations: {uniq_header}')
    print(f'Unique segment relnames: {uniq_segment}')
    print(f'Unique group relnames: {uniq_group}')
    print(f'CSV written to: {out}')


if __name__ == '__main__':
    fire.Fire(collect)
