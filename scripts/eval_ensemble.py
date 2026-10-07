#!/usr/bin/env python3
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

from eval_thresholds import build_model, collect_outputs, evaluate_cached, load_data, parse_thresholds
from model.metrics_utils import is_clashed


def parse_float_list(value):
    items = []
    for part in value.split(','):
        part = part.strip()
        if part:
            items.append(float(part))
    return items


def parse_path_list(value):
    items = []
    for part in value.split(','):
        part = part.strip()
        if part:
            items.append(part)
    return items


def normalize_weights(weights):
    total = sum(weights)
    if total <= 0:
        raise ValueError(f'weights must sum to a positive value, got {weights}')
    return [w / total for w in weights]


def collect_checkpoint_outputs(args, checkpoints, device):
    dataloaders, matrix_segs = load_data(args.model_name, args.dataset_name, args.batch_size, args.num_workers)
    checkpoint_outputs = []
    for checkpoint in checkpoints:
        print(f'Loading checkpoint: {checkpoint}', file=sys.stderr)
        args.checkpoint = checkpoint
        model = build_model(args.model_name, matrix_segs, args).to(device)
        checkpoint_outputs.append(collect_outputs(model, dataloaders, device))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return checkpoint_outputs


def combine_outputs(checkpoint_outputs, weights):
    weights = normalize_weights(weights)
    combined = {}
    for split in checkpoint_outputs[0].keys():
        combined[split] = []
        for batch_idx in range(len(checkpoint_outputs[0][split])):
            weighted_probs = None
            reference = checkpoint_outputs[0][split][batch_idx]
            for model_idx, model_outputs in enumerate(checkpoint_outputs):
                scores = model_outputs[split][batch_idx]['scores'].float().sigmoid()
                if weighted_probs is None:
                    weighted_probs = scores * weights[model_idx]
                else:
                    weighted_probs = weighted_probs + scores * weights[model_idx]
            combined[split].append(
                {
                    'probs': weighted_probs,
                    'ent_target': reference['ent_target'],
                    'word_len': reference['word_len'],
                }
            )
    return combined


def _build_sample_cache_from_probs(probs, ent_target, word_len, min_threshold):
    ent_scores = (probs + probs.transpose(1, 2)) / 2
    batch_cache = []
    for sample_scores, sample_target, sample_len in zip(ent_scores, ent_target, word_len.cpu().tolist()):
        sample_scores = sample_scores[:sample_len, :sample_len]
        max_scores, ent_types = sample_scores.max(dim=-1)
        row_idx, col_idx = torch.triu_indices(sample_len, sample_len)
        span_scores = max_scores[row_idx, col_idx]
        keep = span_scores >= min_threshold
        row_idx = row_idx[keep]
        col_idx = col_idx[keep]
        span_scores = span_scores[keep]
        span_types = ent_types[row_idx, col_idx]
        order = torch.argsort(span_scores, descending=True)
        candidates = [
            (
                float(span_scores[i]),
                int(row_idx[i]),
                int(col_idx[i]),
                int(span_types[i]),
            )
            for i in order
        ]
        batch_cache.append(
            {
                'target': set(map(tuple, sample_target)),
                'candidates': candidates,
            }
        )
    return batch_cache


def build_decode_cache_from_probs(outputs_by_split, thresholds):
    min_threshold = min(thresholds)
    cache = {}
    for split, items in outputs_by_split.items():
        split_cache = []
        for item in items:
            split_cache.extend(
                _build_sample_cache_from_probs(
                    probs=item['probs'],
                    ent_target=item['ent_target'],
                    word_len=item['word_len'],
                    min_threshold=min_threshold,
                )
            )
        cache[split] = split_cache
    return cache


def _decode_candidates(candidates, threshold):
    filtered = []
    for score, start, end, ent_type in candidates:
        if score < threshold:
            break
        chunk = (ent_type, start, end)
        if all(not is_clashed(chunk, existing, allow_nested=True) for existing in filtered):
            filtered.append(chunk)
    return {(start, end, ent_type) for ent_type, start, end in filtered}


def evaluate_ensemble(checkpoint_outputs, checkpoints, thresholds, weight_sets):
    runs = []
    for weights in weight_sets:
        weights = normalize_weights(weights)
        combined = combine_outputs(checkpoint_outputs, weights)
        decode_cache = build_decode_cache_from_probs(combined, thresholds)
        results = evaluate_cached(decode_cache, thresholds)
        best_test = max(results, key=lambda x: x['test_f1'])
        best_dev = max(results, key=lambda x: x['dev_f1'])
        runs.append(
            {
                'weights': weights,
                'checkpoints': checkpoints,
                'results': results,
                'best_test': best_test,
                'best_dev': best_dev,
            }
        )
    return runs


def build_weight_sets(args, checkpoint_count):
    if args.weight_grid:
        if checkpoint_count != 2:
            raise ValueError('--weight-grid currently supports exactly two checkpoints')
        return [[w, 1.0 - w] for w in parse_float_list(args.weight_grid)]
    if args.weights:
        weights = parse_float_list(args.weights)
        if len(weights) != checkpoint_count:
            raise ValueError(f'--weights has {len(weights)} values, but {checkpoint_count} checkpoints were provided')
        return [weights]
    return [[1.0 / checkpoint_count] * checkpoint_count]


def main():
    parser = argparse.ArgumentParser(description='Evaluate probability-averaged ensembles for DiFiNet checkpoints.')
    parser.add_argument('--checkpoints', required=True, help='Comma-separated checkpoint paths.')
    parser.add_argument('--weights', default='', help='Comma-separated weights for one ensemble run.')
    parser.add_argument(
        '--weight-grid',
        default='',
        help='Comma-separated first-checkpoint weights for two-checkpoint sweeps. The second weight is 1-w.',
    )
    parser.add_argument('--dataset-name', default='genia')
    parser.add_argument(
        '--model-name',
        default=os.path.join(PROJECT_ROOT, 'pretrained_models', 'biobert-v1.1'),
    )
    parser.add_argument('--thresholds', default='0.505')
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

    checkpoints = parse_path_list(args.checkpoints)
    if len(checkpoints) < 2:
        raise ValueError('Provide at least two checkpoints for ensemble evaluation')
    thresholds = parse_thresholds(args.thresholds)
    weight_sets = build_weight_sets(args, len(checkpoints))
    device = torch.device(args.device)

    checkpoint_outputs = collect_checkpoint_outputs(args, checkpoints, device)
    runs = evaluate_ensemble(checkpoint_outputs, checkpoints, thresholds, weight_sets)
    payload = {
        'checkpoints': checkpoints,
        'thresholds': thresholds,
        'runs': runs,
        'best_test': max(runs, key=lambda x: x['best_test']['test_f1'])['best_test'],
        'best_dev': max(runs, key=lambda x: x['best_dev']['dev_f1'])['best_dev'],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
