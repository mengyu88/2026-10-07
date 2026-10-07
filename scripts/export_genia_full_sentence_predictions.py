#!/usr/bin/env python3
"""Export complete GENIA predictions using the existing checkpoint and decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from model.model import CNNNer
from model.metrics_utils import decode as decode_spans, _compute_f_rec_pre


def read_jsonlines(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def load_checkpoint(model: CNNNer, checkpoint: Path, label_count: int) -> list[str]:
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    ignored_training_buffers = []
    if 'label_loss_weights' in state:
        if state['label_loss_weights'].shape != (1, 1, 1, label_count):
            raise ValueError('Checkpoint label-loss buffer does not match the label count')
        del state['label_loss_weights']
        ignored_training_buffers.append('label_loss_weights')
    model.load_state_dict(state, strict=True)
    return ignored_training_buffers


def decode(logits: torch.Tensor, length: int, labels: list[str], threshold: float) -> list[dict]:
    probabilities = logits.sigmoid()
    probabilities = (probabilities + probabilities.transpose(0, 1)) / 2
    probabilities = probabilities[:length, :length]
    confidence = probabilities.max(dim=-1).values
    spans = decode_spans(confidence.unsqueeze(0), torch.tensor([length]),
                         allow_nested=True, thres=threshold)[0]
    predictions = []
    for start, end, _ in spans:
        entity_type = int(probabilities[start, end].argmax())
        predictions.append({'start': start, 'end': end + 1,
                            'type': labels[entity_type],
                            'score': float(probabilities[start, end, entity_type])})
    return sorted(predictions, key=lambda item: (item['start'], -item['end'], item['type']))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-name', type=Path, default=PROJECT_ROOT / 'pretrained_models/biobert-v1.1')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'preprocess/outputs/genia')
    parser.add_argument('--threshold', type=float, default=0.48)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--use-snsa', type=int, choices=(0, 1), default=1)
    parser.add_argument('--use-hsr', type=int, choices=(0, 1), default=1)
    parser.add_argument('--sample-indices', type=int, nargs='+')
    parser.add_argument('--pooling-mode', choices=('max', 'legacy_zero_clamped'), default='max')
    parser.add_argument('--batch-order', choices=('original', 'legacy_ascending'), default='original')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(20261002)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = read_jsonlines(args.data_dir / 'train.jsonlines')
    labels = sorted({item['entity_type'] for row in train for item in row['entity_mentions']})
    test = read_jsonlines(args.data_dir / 'test.jsonlines')
    sample_indices = sorted(set(args.sample_indices)) if args.sample_indices is not None else list(range(len(test)))
    if any(sample_index < 0 or sample_index >= len(test) for sample_index in sample_indices):
        parser.error('--sample-indices must refer to existing zero-based test rows')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name), local_files_only=True)
    encoded = []
    token_cache = {}
    requested_indices = set(sample_indices)
    for sample_index, row in enumerate(test):
        input_ids = [tokenizer.cls_token_id]
        indexes = [0]
        for word_index, token in enumerate(row['tokens'], start=1):
            if token not in token_cache:
                token_cache[token] = tokenizer.encode(token, add_special_tokens=False)
            subwords = token_cache[token]
            if not subwords:
                raise ValueError(f'Empty tokenization at sample {sample_index}: {token!r}')
            input_ids.extend(subwords)
            indexes.extend([word_index] * len(subwords))
        input_ids.append(tokenizer.sep_token_id)
        indexes.append(0)
        if len(input_ids) > 400:
            raise ValueError(f'Sample {sample_index} exceeds the existing 400-BPE preprocessing limit')
        encoded.append((sample_index, input_ids, indexes))
    if args.batch_order == 'original':
        order = np.argsort(np.array([len(item[1]) for item in encoded], dtype=int))[::-1]
        encoded = [encoded[int(position)] for position in order]
    else:
        encoded.sort(key=lambda item: len(item[1]))
    model = CNNNer(str(args.model_name), num_ner_tag=len(labels), cnn_dim=400,
                   biaffine_size=200, size_embed_dim=25, logit_drop=0.15,
                   n_layer=2, kernel_size=3, n_head=4, cnn_depth=1,
                   separateness_rate=0.05, theta=1.0, sad_topk=2,
                   use_snsa=bool(args.use_snsa), use_hsr=bool(args.use_hsr),
                   sad_use_rel_bias=True, sad_gate=True, subword_pooling=args.pooling_mode)
    ignored_training_buffers = load_checkpoint(model, args.checkpoint, len(labels))
    model.to(device).eval()
    print(json.dumps({'device': str(device), 'labels': labels, 'sentences': len(sample_indices),
                      'use_snsa': bool(args.use_snsa), 'use_hsr': bool(args.use_hsr),
                      'threshold': args.threshold, 'checkpoint': str(args.checkpoint),
                      'pooling_mode': args.pooling_mode, 'batch_order': args.batch_order}), flush=True)
    results = {}
    with torch.inference_mode():
        for batch_start in range(0, len(encoded), args.batch_size):
            batch = encoded[batch_start:batch_start + args.batch_size]
            if not any(item[0] in requested_indices for item in batch):
                continue
            max_bpe = max(len(item[1]) for item in batch)
            max_words = max(len(test[item[0]]['tokens']) for item in batch)
            input_ids = torch.full((len(batch), max_bpe), tokenizer.pad_token_id, dtype=torch.long, device=device)
            indexes = torch.zeros((len(batch), max_bpe), dtype=torch.long, device=device)
            lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long, device=device)
            matrix = torch.zeros((len(batch), max_words, max_words, len(labels)), device=device)
            for batch_index, (_, ids, word_indexes) in enumerate(batch):
                input_ids[batch_index, :len(ids)] = torch.tensor(ids, device=device)
                indexes[batch_index, :len(ids)] = torch.tensor(word_indexes, device=device)
            raw_words = [test[item[0]]['tokens'] for item in batch]
            scores = model(input_ids=input_ids, bpe_len=lengths, indexes=indexes,
                           matrix=matrix, raw_words=raw_words)['scores'].detach().cpu()
            for batch_index, (sample_index, _, _) in enumerate(batch):
                if sample_index not in requested_indices:
                    continue
                row = test[sample_index]
                gold = sorted([{'start': item['start'], 'end': item['end'], 'type': item['entity_type']}
                               for item in row['entity_mentions']],
                              key=lambda item: (item['start'], -item['end'], item['type']))
                results[sample_index] = {'sample_index': sample_index, 'tokens': row['tokens'],
                                         'gold': gold, 'predictions': decode(scores[batch_index], len(row['tokens']), labels, args.threshold),
                                         'inference_group': [item[0] for item in batch]}
            if batch_start % 200 == 0:
                print(f'Exported {len(results)}/{len(sample_indices)} sentences', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for sample_index in sample_indices:
            handle.write(json.dumps(results[sample_index], ensure_ascii=False) + '\n')
    true_positive = predicted_count = gold_count = 0
    for row in results.values():
        gold = {(entity['start'], entity['end'], entity['type']) for entity in row['gold']}
        predicted = {(entity['start'], entity['end'], entity['type']) for entity in row['predictions']}
        true_positive += len(gold & predicted)
        predicted_count += len(predicted)
        gold_count += len(gold)
    f1, recall, precision = _compute_f_rec_pre(true_positive, gold_count, predicted_count)
    metrics = {'tp': true_positive, 'predicted': predicted_count, 'gold': gold_count,
               'precision': precision, 'recall': recall, 'f1': f1}
    metadata = {'checkpoint': str(args.checkpoint.resolve()), 'model_name': str(args.model_name.resolve()),
                'checkpoint_sha256': file_sha256(args.checkpoint),
                'gold_source': str((args.data_dir / 'test.jsonlines').resolve()), 'threshold': args.threshold,
                'gold_source_sha256': file_sha256(args.data_dir / 'test.jsonlines'),
                'use_snsa': bool(args.use_snsa), 'use_hsr': bool(args.use_hsr),
                'ignored_training_buffers': ignored_training_buffers,
                'sample_indices': sample_indices, 'test_split_sentences': len(test),
                'label_order': labels, 'span_convention': 'zero-based, end-exclusive',
                'decoder': 'sigmoid, symmetric averaging, max type, nested-aware clash filtering',
                'decoder_mode': 'original model.metrics_utils.decode', 'pooling_mode': args.pooling_mode,
                'batch_order': args.batch_order, 'batch_size': args.batch_size, 'precision_mode': 'fp32',
                'metrics': metrics,
                'inference_sources_sha256': {name: file_sha256(PROJECT_ROOT / name) for name in
                                            ('model/model.py', 'model/metrics_utils.py', 'model/cnn_liabrary.py',
                                             'model/pyramid_bfm.py', 'scripts/export_genia_full_sentence_predictions.py')},
                'sentences': len(sample_indices), 'gold_entities': sum(len(row['gold']) for row in results.values()),
                'predicted_entities': sum(len(row['predictions']) for row in results.values())}
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}', flush=True)
    print(json.dumps({'metrics': metrics, 'pooling_mode': args.pooling_mode, 'batch_order': args.batch_order}), flush=True)


if __name__ == '__main__':
    main()
