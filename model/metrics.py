import torch
from fastNLP import Metric
import numpy as np
from .metrics_utils import _compute_f_rec_pre, decode


class NERMetric(Metric):
    def __init__(self, matrix_segs, ent_thres, allow_nested=True):
        super(NERMetric, self).__init__()
        self.register_element('tp', 0, aggregate_method='sum')
        self.register_element('pre', 0, aggregate_method='sum')
        self.register_element('rec', 0, aggregate_method='sum')

        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres

    def update(self, ent_target, scores, word_len):
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class
        ent_scores = (ent_scores + ent_scores.transpose(1, 2))/2
        span_pred = ent_scores.max(dim=-1)[0]

        span_ents = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        for ents, span_ent, ent_pred in zip(ent_target, span_ents, ent_scores.cpu().numpy()):
            pred_ent = set()
            for s, e, l in span_ent:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                if score[ent_type]>=self.ent_thres:
                    pred_ent.add((s, e, ent_type))
            ents = set(map(tuple, ents))
            self.tp += len(ents.intersection(pred_ent))
            self.pre += len(pred_ent)
            self.rec += len(ents)

    def get_metric(self) -> dict:
        f, rec, pre = _compute_f_rec_pre(self.tp, self.rec, self.pre)
        # Keep the historical metric names used by FastNLP's monitor, and
        # additionally expose the underlying sufficient statistics.  The
        # latter make the epoch CSV auditable and allow paper tables to be
        # recomputed without rounding a precision/recall value.
        res = {
            'f': f,
            'rec': rec,
            'pre': pre,
            'tp_count': int(self.tp.get_scalar()),
            'pred_count': int(self.pre.get_scalar()),
            'gold_count': int(self.rec.get_scalar()),
        }
        return res


class PerClassNERMetric(Metric):
    """Exact-span P/R/F1 for every entity class, using the main NER decoder."""
    def __init__(self, matrix_segs, ent_thres, label2idx, allow_nested=True):
        super(PerClassNERMetric, self).__init__()
        assert len(matrix_segs) == 1, "Only support pure entities."
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres
        self.label2idx = {str(label): int(index) for label, index in label2idx.items()}
        self.idx2label = {index: label for label, index in self.label2idx.items()}
        for index in self.idx2label:
            self.register_element(f'tp_{index}', 0, aggregate_method='sum')
            self.register_element(f'pre_{index}', 0, aggregate_method='sum')
            self.register_element(f'rec_{index}', 0, aggregate_method='sum')

    def update(self, ent_target, scores, word_len):
        ent_scores = scores.sigmoid()
        ent_scores = (ent_scores + ent_scores.transpose(1, 2)) / 2
        span_pred = ent_scores.max(dim=-1)[0]
        span_ents = decode(span_pred, word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
        for gold_entities, predicted_spans, predicted_scores in zip(ent_target, span_ents, ent_scores.cpu().numpy()):
            predicted = set()
            for start, end, _ in predicted_spans:
                score = predicted_scores[start, end]
                label_index = int(score.argmax())
                if score[label_index] >= self.ent_thres:
                    predicted.add((start, end, label_index))
            gold = set(map(tuple, gold_entities))
            for label_index in self.idx2label:
                gold_for_label = {entity for entity in gold if entity[2] == label_index}
                pred_for_label = {entity for entity in predicted if entity[2] == label_index}
                for name, increment in (
                    (f'tp_{label_index}', len(gold_for_label.intersection(pred_for_label))),
                    (f'pre_{label_index}', len(pred_for_label)),
                    (f'rec_{label_index}', len(gold_for_label)),
                ):
                    setattr(self, name, getattr(self, name) + increment)

    def get_metric(self) -> dict:
        result = {}
        for label_index, label in self.idx2label.items():
            f1, recall, precision = _compute_f_rec_pre(
                getattr(self, f'tp_{label_index}'),
                getattr(self, f'rec_{label_index}'),
                getattr(self, f'pre_{label_index}'))
            prefix = label.lower()
            result[f'{prefix}_f'] = f1
            result[f'{prefix}_rec'] = recall
            result[f'{prefix}_pre'] = precision
            result[f'{prefix}_tp_count'] = int(getattr(self, f'tp_{label_index}').get_scalar())
            result[f'{prefix}_pred_count'] = int(getattr(self, f'pre_{label_index}').get_scalar())
            result[f'{prefix}_gold_count'] = int(getattr(self, f'rec_{label_index}').get_scalar())
        return result
