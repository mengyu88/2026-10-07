#!/usr/bin/env python3
"""Report exact-span precision, recall and F1 for every entity type."""
import argparse
import json
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from eval_thresholds import build_model, collect_outputs, load_data
from model.metrics_utils import _compute_f_rec_pre, is_clashed


def decode_candidates(candidates, threshold):
    selected = []
    for score, start, end, label_id in candidates:
        if score < threshold:
            break
        candidate = (label_id, start, end)
        if all(not is_clashed(candidate, item, allow_nested=True) for item in selected):
            selected.append(candidate)
    return {(start, end, label_id) for label_id, start, end in selected}


def build_cache(outputs, threshold):
    cache = []
    for item in outputs:
        scores = item['scores'].sigmoid()
        scores = (scores + scores.transpose(1, 2)) / 2
        for sample_scores, target, word_len in zip(scores, item['ent_target'], item['word_len'].tolist()):
            sample_scores = sample_scores[:word_len, :word_len]
            max_scores, label_ids = sample_scores.max(dim=-1)
            starts, ends = torch.triu_indices(word_len, word_len)
            candidate_scores = max_scores[starts, ends]
            keep = candidate_scores >= threshold
            starts, ends = starts[keep], ends[keep]
            candidate_scores, label_ids = candidate_scores[keep], label_ids[starts, ends]
            order = torch.argsort(candidate_scores, descending=True)
            candidates = [
                (float(candidate_scores[i]), int(starts[i]), int(ends[i]), int(label_ids[i]))
                for i in order
            ]
            cache.append((set(map(tuple, target)), candidates))
    return cache


def read_label_names(dataset_name):
    labels = set()
    path = os.path.join(PROJECT_ROOT, 'preprocess', 'outputs', dataset_name, 'train.jsonlines')
    with open(path, encoding='utf-8') as file:
        for line in file:
            for ent in json.loads(line)['entity_mentions']:
                labels.add(ent['entity_type'])
    return sorted(labels)


def evaluate_by_class(cache, label_names, threshold):
    counts = {label: {'tp': 0, 'pred': 0, 'gold': 0} for label in label_names}
    for gold, candidates in cache:
        prediction = decode_candidates(candidates, threshold)
        for start, end, label_id in prediction:
            counts[label_names[label_id]]['pred'] += 1
        for start, end, label_id in gold:
            counts[label_names[label_id]]['gold'] += 1
        for start, end, label_id in gold.intersection(prediction):
            counts[label_names[label_id]]['tp'] += 1

    rows = []
    total = {'tp': 0, 'pred': 0, 'gold': 0}
    for label in label_names:
        item = counts[label]
        f1, recall, precision = _compute_f_rec_pre(item['tp'], item['gold'], item['pred'])
        rows.append({
            'entity_type': label, 'precision': precision, 'recall': recall, 'f1': f1,
            'tp': item['tp'], 'predicted': item['pred'], 'gold': item['gold'],
        })
        for key in total:
            total[key] += item[key]
    f1, recall, precision = _compute_f_rec_pre(total['tp'], total['gold'], total['pred'])
    rows.append({
        'entity_type': 'Overall', 'precision': precision, 'recall': recall, 'f1': f1,
        'tp': total['tp'], 'predicted': total['pred'], 'gold': total['gold'],
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Per-class SPSR-Net NER evaluation.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-name', default='food')
    parser.add_argument('--model-name', default=os.path.join(PROJECT_ROOT, 'pretrained_models', 'bert-base-chinese'))
    parser.add_argument('--threshold', type=float, default=0.48)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='')
    parser.add_argument('--cnn-dim', type=int, default=400)
    parser.add_argument('--biaffine-size', type=int, default=200)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--cnn-depth', type=int, default=1)
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--logit-drop', type=float, default=0.15)
    parser.add_argument('--size-embed-dim', type=int, default=25)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--separateness-rate', type=float, default=0.05)
    parser.add_argument('--theta', type=float, default=1.0)
    parser.add_argument('--sad-topk', type=int, default=2)
    parser.add_argument('--sad-attn-dim', type=int, default=None)
    parser.add_argument('--use-snsa', type=int, default=1)
    parser.add_argument('--use-hsr', type=int, default=1)
    parser.add_argument('--sad-use-rel-bias', type=int, default=1)
    parser.add_argument('--sad-gate', type=int, default=1)
    args = parser.parse_args()

    label_names = read_label_names(args.dataset_name)
    dataloaders, matrix_segs = load_data(args.model_name, args.dataset_name, args.batch_size, args.num_workers)
    model = build_model(args.model_name, matrix_segs, args).to(torch.device(args.device))
    outputs = collect_outputs(model, {'test': dataloaders['test']}, torch.device(args.device))['test']
    rows = evaluate_by_class(build_cache(outputs, args.threshold), label_names, args.threshold)
    payload = {'checkpoint': args.checkpoint, 'dataset': args.dataset_name, 'threshold': args.threshold, 'rows': rows}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
