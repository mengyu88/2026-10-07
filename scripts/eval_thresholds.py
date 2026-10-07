#!/usr/bin/env python3
import argparse
import json
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from fastNLP import SortedSampler, prepare_torch_dataloader

from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
from model.metrics_utils import _compute_f_rec_pre, is_clashed
from model.model import CNNNer


def densify(x):
    return x.todense().astype(np.float32)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return value
    return value


def load_data(model_name, dataset_name, batch_size, num_workers, data_dir=''):
    """Load evaluation splits, optionally from an immutable experiment split."""
    paths = data_dir or os.path.join(PROJECT_ROOT, 'preprocess', 'outputs', dataset_name)
    pipe = SpanNerPipe(model_name=model_name)
    data_bundle = pipe.process_from_file(paths)
    data_bundle.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')
    matrix_segs = pipe.matrix_segs

    dataloaders = {}
    for name, dataset in data_bundle.iter_datasets():
        dataset.set_pad(
            'matrix',
            pad_fn=Torch3DMatrixPadder(
                pad_val=dataset.collator.input_fields['matrix']['pad_val'],
                num_class=matrix_segs['ent'],
                batch_size=batch_size,
            ),
        )
        if name in ('dev', 'test'):
            dataloaders[name] = prepare_torch_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=SortedSampler(dataset, 'input_ids'),
                pin_memory=torch.cuda.is_available(),
                shuffle=False,
            )
    return dataloaders, matrix_segs


def build_model(model_name, matrix_segs, args):
    model = CNNNer(
        model_name,
        num_ner_tag=matrix_segs['ent'],
        cnn_dim=args.cnn_dim,
        biaffine_size=args.biaffine_size,
        size_embed_dim=args.size_embed_dim,
        logit_drop=args.logit_drop,
        n_layer=args.n_layer,
        kernel_size=args.kernel_size,
        n_head=args.n_head,
        cnn_depth=args.cnn_depth,
        separateness_rate=args.separateness_rate,
        theta=args.theta,
        sad_topk=args.sad_topk,
        sad_attn_dim=args.sad_attn_dim,
        use_snsa=bool(args.use_snsa),
        use_hsr=bool(args.use_hsr),
        sad_use_rel_bias=bool(args.sad_use_rel_bias),
        sad_gate=bool(args.sad_gate),
    )
    state = torch.load(args.checkpoint, map_location='cpu')
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            json.dumps(
                {
                    'missing_keys': load_result.missing_keys,
                    'unexpected_keys': load_result.unexpected_keys,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    return model


@torch.no_grad()
def collect_outputs(model, dataloaders, device):
    model.eval()
    outputs_by_split = {}
    for split, dataloader in dataloaders.items():
        split_outputs = []
        for batch in dataloader:
            batch_on_device = move_to_device(batch, device)
            outputs = model(
                input_ids=batch_on_device['input_ids'],
                bpe_len=batch_on_device['bpe_len'],
                indexes=batch_on_device['indexes'],
                matrix=batch_on_device['matrix'],
                raw_words=batch_on_device['raw_words'],
            )
            split_outputs.append(
                {
                    'scores': outputs['scores'].detach().cpu(),
                    'ent_target': batch['ent_target'],
                    'word_len': batch['word_len'].detach().cpu()
                    if torch.is_tensor(batch['word_len'])
                    else batch['word_len'],
                }
            )
        outputs_by_split[split] = split_outputs
    return outputs_by_split


def _build_sample_cache(scores, ent_target, word_len, min_threshold):
    ent_scores = scores.sigmoid()
    ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
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


def build_decode_cache(outputs_by_split, thresholds):
    min_threshold = min(thresholds)
    cache = {}
    for split, items in outputs_by_split.items():
        split_cache = []
        for item in items:
            split_cache.extend(
                _build_sample_cache(
                    scores=item['scores'],
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


def evaluate_cached(decode_cache, thresholds):
    results = []
    for threshold in thresholds:
        row = {'threshold': threshold}
        for split in ('dev', 'test'):
            tp = 0
            pred_count = 0
            gold_count = 0
            for item in decode_cache[split]:
                pred = _decode_candidates(item['candidates'], threshold)
                gold = item['target']
                tp += len(gold.intersection(pred))
                pred_count += len(pred)
                gold_count += len(gold)
            f, rec, pre = _compute_f_rec_pre(tp, gold_count, pred_count)
            row[f'{split}_f1'] = f
            row[f'{split}_rec'] = rec
            row[f'{split}_pre'] = pre
        results.append(row)
    return results


def parse_thresholds(value):
    thresholds = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    return thresholds


def main():
    parser = argparse.ArgumentParser(description='Evaluate a DiFiNet checkpoint across entity thresholds.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-name', default='genia')
    parser.add_argument('--data-dir', default='',
                        help='Optional directory containing train/dev/test.jsonlines.')
    parser.add_argument(
        '--model-name',
        default=os.path.join(PROJECT_ROOT, 'pretrained_models', 'biobert-v1.1'),
    )
    parser.add_argument('--thresholds', default='0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65')
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

    thresholds = parse_thresholds(args.thresholds)
    device = torch.device(args.device)
    dataloaders, matrix_segs = load_data(
        args.model_name, args.dataset_name, args.batch_size, args.num_workers, args.data_dir
    )
    model = build_model(args.model_name, matrix_segs, args).to(device)
    outputs_by_split = collect_outputs(model, dataloaders, device)
    decode_cache = build_decode_cache(outputs_by_split, thresholds)
    results = evaluate_cached(decode_cache, thresholds)

    best_test = max(results, key=lambda x: x['test_f1'])
    best_dev = max(results, key=lambda x: x['dev_f1'])
    payload = {
        'checkpoint': args.checkpoint,
        'thresholds': thresholds,
        'results': results,
        'best_test': best_test,
        'best_dev': best_dev,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
