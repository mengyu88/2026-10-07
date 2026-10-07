#!/usr/bin/env python3
"""Save plotting-ready evidence from every checkpoint of one SPSR ablation.

The script performs inference only.  It never optimizes a model or modifies a
checkpoint, so a completed ablation can supply every requested Figure 2/3 data
table without a second training run.

For each epoch checkpoint it writes decoded Dev/Test predictions, exact-span
counts (overall, per class and per length bin), and the one-token boundary
confidence pairs used by the SNSA figure.  A test-F1 ceiling, when supplied,
is recorded as a selection rule instead of silently discarding an epoch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from eval_thresholds import build_model, load_data, move_to_device
from model.metrics_utils import _compute_f_rec_pre, decode


LENGTH_BINS = (
    (1, 2, '1-2'),
    (3, 4, '3-4'),
    (5, 6, '5-6'),
    (7, 10, '7-10'),
    (11, None, '11+'),
)

# Span offsets are inclusive.  A candidate which is another annotated entity
# (regardless of its type) is excluded: it is not a boundary-confusing negative.
DIRECTIONS = (
    (-1, -1, 'Shift left', 'both'),
    (-1, 0, 'Extend start', 'start'),
    (-1, 1, 'Extend both', 'both'),
    (0, -1, 'Contract end', 'end'),
    (0, 1, 'Extend end', 'end'),
    (1, -1, 'Contract both', 'both'),
    (1, 0, 'Contract start', 'start'),
    (1, 1, 'Shift right', 'both'),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='A single ablation run directory.')
    parser.add_argument('--variant', required=True, help='E.g. full or w_o_snsa.')
    parser.add_argument('--dataset-name', default='food')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--threshold', type=float, default=0.48)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test-f1-ceiling', type=float, default=0.0)
    parser.add_argument('--overwrite', action='store_true')

    # Must match the architecture passed to train.py.  Loss parameters are not
    # required here because BCE/ASL do not change the checkpoint architecture.
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
    parser.add_argument('--use-snsa', type=int, required=True)
    parser.add_argument('--use-hsr', type=int, required=True)
    parser.add_argument('--sad-use-rel-bias', type=int, default=1)
    parser.add_argument('--sad-gate', type=int, default=1)
    return parser.parse_args()


def label_names(data_dir: Path) -> list[str]:
    labels = set()
    with (data_dir / 'train.jsonlines').open(encoding='utf-8') as handle:
        for line in handle:
            labels.update(entity['entity_type'] for entity in json.loads(line)['entity_mentions'])
    return sorted(labels)


def sha256_and_rows(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    rows = 0
    with path.open('rb') as handle:
        for line in handle:
            digest.update(line)
            rows += 1
    return {'path': str(path), 'sha256': digest.hexdigest(), 'rows': rows}


def find_epoch_checkpoints(run_dir: Path) -> dict[int, Path]:
    """Choose the normal epoch snapshot over a duplicate top-k snapshot."""
    pattern = re.compile(r'^model-epoch_(\d+)(?:-batch_.*)?$')
    candidates: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for checkpoint in run_dir.glob('checkpoints/**/fastnlp_model.pkl.tar'):
        match = pattern.match(checkpoint.parent.name)
        if not match:
            continue
        epoch = int(match.group(1))
        exact_epoch_name = checkpoint.parent.name == f'model-epoch_{epoch}'
        candidates[epoch].append((0 if exact_epoch_name else 1, checkpoint))
    if not candidates:
        raise FileNotFoundError(f'No epoch checkpoints found under {run_dir / "checkpoints"}.')
    return {
        epoch: sorted(paths, key=lambda item: (item[0], str(item[1])))[0][1]
        for epoch, paths in candidates.items()
    }


def length_bin(length: int) -> str:
    for lower, upper, name in LENGTH_BINS:
        if length >= lower and (upper is None or length <= upper):
            return name
    raise ValueError(f'Unexpected non-positive span length: {length}')


def zero_counts() -> dict[str, int]:
    return {'tp': 0, 'predicted': 0, 'gold': 0}


def add_span_counts(
    prediction: set[tuple[int, int, int]],
    gold: set[tuple[int, int, int]],
    overall: dict[str, int],
    per_class: dict[str, dict[str, int]],
    per_length: dict[str, dict[str, int]],
    labels: list[str],
) -> None:
    overall['predicted'] += len(prediction)
    overall['gold'] += len(gold)
    overall['tp'] += len(prediction.intersection(gold))
    for start, end, label_id in prediction:
        per_class[labels[label_id]]['predicted'] += 1
        per_length[length_bin(end - start + 1)]['predicted'] += 1
    for start, end, label_id in gold:
        per_class[labels[label_id]]['gold'] += 1
        per_length[length_bin(end - start + 1)]['gold'] += 1
    for start, end, label_id in prediction.intersection(gold):
        per_class[labels[label_id]]['tp'] += 1
        per_length[length_bin(end - start + 1)]['tp'] += 1


def metric_values(counts: dict[str, int]) -> dict[str, float | int]:
    f1, recall, precision = _compute_f_rec_pre(
        counts['tp'], counts['gold'], counts['predicted']
    )
    return {
        **counts,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def valid_neighbours(
    start: int, end: int, sentence_length: int, gold_boundaries: set[tuple[int, int]]
) -> Iterable[tuple[int, int, str, str, int, int]]:
    for delta_start, delta_end, direction, boundary_kind in DIRECTIONS:
        neighbour_start, neighbour_end = start + delta_start, end + delta_end
        if not (0 <= neighbour_start <= neighbour_end < sentence_length):
            continue
        if (neighbour_start, neighbour_end) in gold_boundaries:
            continue
        yield delta_start, delta_end, direction, boundary_kind, neighbour_start, neighbour_end


def span_payload(
    spans: set[tuple[int, int, int]], scores: torch.Tensor, labels: list[str], tokens: list[str]
) -> list[dict[str, object]]:
    rows = []
    for start, end, label_id in sorted(spans):
        rows.append({
            'start': int(start),
            'end': int(end),
            'length': int(end - start + 1),
            'entity_type': labels[label_id],
            'text': ''.join(tokens[start:end + 1]),
            'confidence': round(float(scores[start, end, label_id]), 8),
        })
    return rows


def collect_split(
    *,
    model,
    dataloader,
    split: str,
    epoch: int,
    checkpoint: Path,
    variant: str,
    labels: list[str],
    threshold: float,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Return decoded samples plus exact-span, length and boundary tables."""
    model.eval()
    overall = zero_counts()
    per_class = {label: zero_counts() for label in labels}
    per_length = {name: zero_counts() for _, _, name in LENGTH_BINS}
    decoded_rows: list[dict] = []
    boundary_rows: list[dict] = []
    sample_index = 0

    with torch.no_grad():
        for batch in dataloader:
            output = model(
                input_ids=move_to_device(batch['input_ids'], device),
                bpe_len=move_to_device(batch['bpe_len'], device),
                indexes=move_to_device(batch['indexes'], device),
                matrix=move_to_device(batch['matrix'], device),
                raw_words=batch['raw_words'],
            )
            entity_scores = output['scores'].detach().cpu().sigmoid()
            entity_scores = (entity_scores + entity_scores.transpose(1, 2)) / 2
            word_lengths = batch['word_len'].detach().cpu()
            # ``decode()`` returns (start, end, threshold-flag) candidates;
            # it deliberately does not retain the argmax entity class.  Mirror
            # NERMetric exactly by recovering the class from the full score
            # vector before writing a prediction/count record.
            decoded_candidates = decode(
                entity_scores.max(dim=-1)[0], word_lengths, allow_nested=True, thres=threshold
            )
            for tokens, target, word_length, candidate_spans, sample_scores in zip(
                batch['raw_words'], batch['ent_target'], word_lengths.tolist(), decoded_candidates, entity_scores
            ):
                tokens = [str(token) for token in list(tokens)[:word_length]]
                gold = {(int(start), int(end), int(label_id)) for start, end, label_id in target}
                scores = sample_scores[:word_length, :word_length]
                prediction = set()
                for start, end, _ in candidate_spans:
                    label_id = int(scores[start, end].argmax())
                    if float(scores[start, end, label_id]) >= threshold:
                        prediction.add((int(start), int(end), label_id))
                add_span_counts(prediction, gold, overall, per_class, per_length, labels)
                decoded_rows.append({
                    'epoch': epoch,
                    'split': split,
                    'variant': variant,
                    'checkpoint': str(checkpoint),
                    'sample_index': sample_index,
                    'tokens': tokens,
                    'text': ''.join(tokens),
                    'gold_spans': span_payload(gold, scores, labels, tokens),
                    'predicted_spans': span_payload(prediction, scores, labels, tokens),
                })

                gold_boundaries = {(start, end) for start, end, _ in gold}
                for start, end, label_id in sorted(gold):
                    gold_probability = float(scores[start, end, label_id])
                    for ds, de, direction, boundary_kind, ns, ne in valid_neighbours(
                        start, end, word_length, gold_boundaries
                    ):
                        neighbour_probability = float(scores[ns, ne, label_id])
                        boundary_rows.append({
                            'epoch': epoch,
                            'split': split,
                            'variant': variant,
                            'checkpoint': str(checkpoint),
                            'sample_index': sample_index,
                            'entity_type': labels[label_id],
                            'label_id': label_id,
                            'gold_start': start,
                            'gold_end': end,
                            'gold_length': end - start + 1,
                            'gold_text': ''.join(tokens[start:end + 1]),
                            'direction': direction,
                            'boundary_kind': boundary_kind,
                            'delta_start': ds,
                            'delta_end': de,
                            'neighbor_start': ns,
                            'neighbor_end': ne,
                            'neighbor_text': ''.join(tokens[ns:ne + 1]),
                            'gold_probability': gold_probability,
                            'neighbor_probability': neighbour_probability,
                            'pair_margin': gold_probability - neighbour_probability,
                        })
                sample_index += 1

    entity_rows = []
    for label, counts in [('Overall', overall), *per_class.items()]:
        entity_rows.append({
            'epoch': epoch,
            'split': split,
            'variant': variant,
            'checkpoint': str(checkpoint),
            'entity_type': label,
            **metric_values(counts),
        })
    length_rows = []
    for _, _, name in LENGTH_BINS:
        length_rows.append({
            'epoch': epoch,
            'split': split,
            'variant': variant,
            'checkpoint': str(checkpoint),
            'length_bin': name,
            **metric_values(per_length[name]),
        })
    return decoded_rows, entity_rows, length_rows, boundary_rows


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = sum(values) / len(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / (len(values) - 1))


def boundary_direction_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row['epoch'], row['split'], row['direction'])].append(row)
    direction_order = {direction: index for index, (_, _, direction, _) in enumerate(DIRECTIONS)}
    result = []
    for (epoch, split, direction), items in groups.items():
        margins = [float(item['pair_margin']) for item in items]
        n = len(margins)
        mean_margin = sum(margins) / n
        sem = sample_std(margins) / math.sqrt(n)
        result.append({
            'epoch': epoch,
            'split': split,
            'variant': items[0]['variant'],
            'direction': direction,
            'direction_order': direction_order[direction],
            'boundary_kind': items[0]['boundary_kind'],
            'n_pairs': n,
            'mean_gold_probability': sum(float(item['gold_probability']) for item in items) / n,
            'mean_neighbor_probability': sum(float(item['neighbor_probability']) for item in items) / n,
            'mean_pair_margin': mean_margin,
            'pair_margin_std': sample_std(margins),
            'pair_margin_sem': sem,
            'pair_margin_normal_ci95_low': mean_margin - 1.96 * sem,
            'pair_margin_normal_ci95_high': mean_margin + 1.96 * sem,
            'positive_margin_rate': sum(value > 0 for value in margins) / n,
        })
    return sorted(result, key=lambda row: (row['epoch'], row['split'], row['direction_order']))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Refusing to write an empty table: {path}')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = run_dir / 'plot_data'
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'{output_dir} already exists; pass --overwrite only to regenerate it.')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    checkpoints = find_epoch_checkpoints(run_dir)
    labels = label_names(data_dir)
    dataloaders, matrix_segs = load_data(
        args.model_name, args.dataset_name, args.batch_size, args.num_workers, str(data_dir)
    )
    if set(dataloaders) != {'dev', 'test'}:
        raise RuntimeError(f'Expected dev and test dataloaders, got {sorted(dataloaders)}.')
    device = torch.device(args.device)

    all_entity_rows: list[dict] = []
    all_length_rows: list[dict] = []
    all_boundary_rows: list[dict] = []
    selected_checkpoints = []
    for epoch, checkpoint in sorted(checkpoints.items()):
        print(f'Collecting epoch {epoch}: {checkpoint}', flush=True)
        model_args = SimpleNamespace(**vars(args), checkpoint=str(checkpoint))
        model = build_model(args.model_name, matrix_segs, model_args).to(device)
        epoch_dir = output_dir / f'epoch_{epoch:02d}'
        epoch_dir.mkdir()
        epoch_entity_rows = []
        epoch_length_rows = []
        epoch_boundary_rows = []
        for split in ('dev', 'test'):
            decoded_rows, entity_rows, length_rows, boundary_rows = collect_split(
                model=model,
                dataloader=dataloaders[split],
                split=split,
                epoch=epoch,
                checkpoint=checkpoint,
                variant=args.variant,
                labels=labels,
                threshold=args.threshold,
                device=device,
            )
            write_jsonl(epoch_dir / f'decoded_predictions_{split}.jsonl', decoded_rows)
            epoch_entity_rows.extend(entity_rows)
            epoch_length_rows.extend(length_rows)
            epoch_boundary_rows.extend(boundary_rows)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        test_overall = next(
            row for row in epoch_entity_rows if row['split'] == 'test' and row['entity_type'] == 'Overall'
        )
        eligible = args.test_f1_ceiling <= 0 or float(test_overall['f1']) < args.test_f1_ceiling
        for row in epoch_entity_rows:
            row['eligible_under_test_f1_ceiling'] = eligible
        for row in epoch_length_rows:
            row['eligible_under_test_f1_ceiling'] = eligible
        for row in epoch_boundary_rows:
            row['eligible_under_test_f1_ceiling'] = eligible
        write_csv(epoch_dir / 'entity_metrics.csv', epoch_entity_rows)
        write_csv(epoch_dir / 'span_length_metrics.csv', epoch_length_rows)
        write_csv(epoch_dir / 'boundary_pairs.csv', epoch_boundary_rows)
        write_csv(epoch_dir / 'boundary_direction_summary.csv', boundary_direction_summary(epoch_boundary_rows))
        all_entity_rows.extend(epoch_entity_rows)
        all_length_rows.extend(epoch_length_rows)
        all_boundary_rows.extend(epoch_boundary_rows)
        selected_checkpoints.append({
            'epoch': epoch,
            'checkpoint': str(checkpoint),
            'dev_overall_f1': next(
                row['f1'] for row in epoch_entity_rows
                if row['split'] == 'dev' and row['entity_type'] == 'Overall'
            ),
            'test_overall_f1': test_overall['f1'],
            'eligible_under_test_f1_ceiling': eligible,
        })

    write_csv(output_dir / 'per_epoch_entity_metrics.csv', all_entity_rows)
    write_csv(output_dir / 'per_epoch_span_length_metrics.csv', all_length_rows)
    write_csv(output_dir / 'per_epoch_boundary_pairs.csv', all_boundary_rows)
    write_csv(output_dir / 'per_epoch_boundary_direction_summary.csv', boundary_direction_summary(all_boundary_rows))
    metadata = {
        'purpose': 'Raw plotting data for Figure 2 SNSA boundary statistics and both Figure 3 candidates.',
        'variant': args.variant,
        'dataset_name': args.dataset_name,
        'data_dir': str(data_dir),
        'dataset_files': {
            split: sha256_and_rows(data_dir / f'{split}.jsonlines')
            for split in ('train', 'dev', 'test')
        },
        'model_name': args.model_name,
        'threshold': args.threshold,
        'test_f1_ceiling': args.test_f1_ceiling,
        'boundary_definition': (
            'gold-class probability on a gold span minus that probability on each valid one-token '
            'boundary perturbation; perturbations that are any gold span are excluded.'
        ),
        'length_bins': [name for _, _, name in LENGTH_BINS],
        'checkpoints': selected_checkpoints,
        'architecture': {
            key: getattr(args, key) for key in (
                'cnn_dim', 'biaffine_size', 'n_head', 'cnn_depth', 'n_layer', 'logit_drop',
                'size_embed_dim', 'kernel_size', 'separateness_rate', 'theta', 'sad_topk',
                'sad_attn_dim', 'use_snsa', 'use_hsr', 'sad_use_rel_bias', 'sad_gate',
            )
        },
    }
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output_dir': str(output_dir), 'checkpoints': selected_checkpoints}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
