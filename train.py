import csv
import json
import os
import warnings
import argparse
# from torchstat import stat
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import random
# from thop import profile
from fastNLP.core.callbacks.topk_saver import TopkSaver
from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import print
from fastNLP import Trainer, Callback
from fastNLP import TorchGradClipCallback
from fastNLP import FitlogCallback, CheckpointCallback, TorchGradClipCallback, EarlyStopCallback
from fastNLP.core.utils.exceptions import EarlyStopException
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog

# fitlog.debug()

from model.model import CNNNer
from model.metrics import NERMetric, PerClassNERMetric
from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
from callbacks import AdversarialTrainingCallback

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--encoder_lr', default=2e-5, type=float)
parser.add_argument('-b', '--batch_size', default=24, type=int)
parser.add_argument('-n', '--n_epochs', default=50, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('-d', '--dataset_name', default='ace2004', type=str) # ace2005 genia ace2004
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--data_dir', default='', type=str,
                    help='Optional directory containing train/dev/test.jsonlines. '
                         'Use this for a reproducible alternative split without replacing the default data.')
parser.add_argument('--ent_thres', default=0.5, type=float)
parser.add_argument('--cnn_depth', default=1, type=int)
parser.add_argument('--cnn_dim', default=120, type=int)
parser.add_argument('--num', default=1, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_layer', default=1, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)
parser.add_argument('--separateness_rate', default=5, type=int)
parser.add_argument('--theta', default=1, type=float)
parser.add_argument('--loss_theta', default=1, type=float)
parser.add_argument('--loss_type', default='bce', choices=['bce', 'asl'], type=str)
parser.add_argument('--asl_gamma_pos', default=0.0, type=float)
parser.add_argument('--asl_gamma_neg', default=3.0, type=float)
parser.add_argument('--asl_clip', default=0.05, type=float)
parser.add_argument('--sad_topk', default=2, type=int)
parser.add_argument('--sad_attn_dim', default=None, type=int)
parser.add_argument('--use_snsa', default=1, type=int)
parser.add_argument('--use_hsr', default=1, type=int)
parser.add_argument('--sad_use_rel_bias', default=1, type=int)
parser.add_argument('--sad_gate', default=1, type=int)
parser.add_argument('--init_from_checkpoint', default='', type=str)
parser.add_argument('--adv_type', default='pgd', choices=['none', 'pgd'], type=str)
parser.add_argument('--adv_epsilon', default=1.0, type=float)
parser.add_argument('--adv_alpha', default=0.3, type=float)
parser.add_argument('--adv_k', default=3, type=int)
parser.add_argument('--adv_emb_name', default='word_embeddings', type=str)
parser.add_argument('--adv_loss_weight', default=1.0, type=float)
parser.add_argument('--adv_warmup_ratio', default=0.0, type=float)
parser.add_argument('--adv_every_n_steps', default=1, type=int)
parser.add_argument('--adv_random_start', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--fp16_init_scale', default=1.0, type=float)
parser.add_argument('--evaluate_every', default=-1, type=int)
parser.add_argument('--num_train_batch_per_epoch', default=-1, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--early_stop_patience', default=0, type=int)
parser.add_argument('--early_stop_monitor', default='f#f#dev', type=str)
parser.add_argument('--checkpoint_monitor', default='f#f#test', type=str)
parser.add_argument('--epoch_metrics_path', default='', type=str,
                    help='Optional CSV path for per-epoch dev/test overall and per-class metrics.')
parser.add_argument('--stop_at_test_f1', default=0.0, type=float,
                    help='If positive, stop before the next epoch once test F1 reaches this ceiling. '
                         'The ceiling epoch is retained as an audit record but must not be selected.')

args = parser.parse_args()
dataset_name = args.dataset_name
def _resolve_default_model_name(dataset_name):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.join(project_dir, 'pretrained_models')
    model_candidates = []
    env_override = None
    hf_fallback = None

    if 'genia' in dataset_name:
        env_override = os.environ.get('DIFINET_GENIA_MODEL')
        model_candidates = [
            os.path.join(model_root, 'biobert-v1.1'),
            os.path.join(model_root, 'biomedbert-base-uncased-abstract-fulltext'),
            os.path.join(model_root, 'bert-base-cased'),
        ]
        hf_fallback = 'dmis-lab/biobert-v1.1'
    elif dataset_name == 'food':
        env_override = os.environ.get('DIFINET_FOOD_MODEL')
        model_candidates = [
            os.path.join(model_root, 'bert-base-chinese'),
            '/root/.cache/modelscope/hub/models/AI-ModelScope/bert-base-chinese',
        ]
        hf_fallback = 'bert-base-chinese'
    elif dataset_name == 'conll03':
        env_override = os.environ.get('DIFINET_CONLL_MODEL')
        model_candidates = [
            os.path.join(model_root, 'bert-large-cased'),
            os.path.join(model_root, 'bert-base-cased'),
        ]
        hf_fallback = 'bert-large-cased'
    elif dataset_name in ('ace2004', 'ace2005'):
        env_override = os.environ.get('DIFINET_ACE_MODEL')
        model_candidates = [os.path.join(model_root, 'roberta-base')]
        hf_fallback = 'roberta-base'
    else:
        raise RuntimeError(f'Unsupported dataset_name={dataset_name}')

    if env_override:
        return env_override
    for local_dir in model_candidates:
        if os.path.isdir(local_dir):
            return local_dir
    return hf_fallback


if args.model_name is None:
    args.model_name = _resolve_default_model_name(args.dataset_name)

model_name = args.model_name
n_head = args.n_head
######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-4
size_embed_dim = 25
ent_thres = args.ent_thres
kernel_size = 3
######hyper

# Allow long-running jobs to place logs/checkpoints on a dedicated volume.
# Defaults preserve the original experiment layout for all existing commands.
fitlog.set_log_dir(os.environ.get('SPSR_LOG_DIR', 'logs/'))

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
def seed_torch(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False


seed = fitlog.set_rng_seed(rng_seed=args.seed)
seed_torch(args.seed)
os.environ['FASTNLP_GLOBAL_SEED'] = str(seed)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name, data_dir=''):
    # 以下是我们自己的数据
    if data_dir:
        paths = data_dir
    elif dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    elif dataset_name == 'food':
        paths = 'preprocess/outputs/food'
    elif dataset_name == 'conll03':
        paths = 'preprocess/outputs/conll03'
    else:
        raise RuntimeError("Does not support.")
    pipe = SpanNerPipe(model_name=model_name)
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs


dl, matrix_segs = get_data(dataset_name, model_name, args.data_dir)


def densify(x):
    x = x.todense().astype(np.float32)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

print(dl)
label2idx = getattr(dl, 'ner_vocab') if hasattr(dl, 'ner_vocab') else getattr(dl, 'label2idx')
print(f"{len(label2idx)} labels: {label2idx}, matrix_segs:{matrix_segs}")

dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=args.batch_size))

    if name == 'train':
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=args.batch_size,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)

    else:
        _dl = prepare_torch_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                       sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=False)
    dls[name] = _dl

model = CNNNer(model_name, num_ner_tag=matrix_segs['ent'], cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
               size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,n_layer=args.n_layer,
               kernel_size=kernel_size, n_head=n_head, cnn_depth=args.cnn_depth,
               separateness_rate=args.separateness_rate/100, theta=args.theta,
               sad_topk=args.sad_topk, sad_attn_dim=args.sad_attn_dim,
               use_snsa=bool(args.use_snsa),
               use_hsr=bool(args.use_hsr),
               sad_use_rel_bias=bool(args.sad_use_rel_bias), sad_gate=bool(args.sad_gate),
               loss_type=args.loss_type,
               asl_gamma_pos=args.asl_gamma_pos, asl_gamma_neg=args.asl_gamma_neg,
               asl_clip=args.asl_clip)

if args.init_from_checkpoint:
    if not os.path.exists(args.init_from_checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.init_from_checkpoint}')
    checkpoint_state = torch.load(args.init_from_checkpoint, map_location='cpu')
    load_result = model.load_state_dict(checkpoint_state, strict=False)
    print(
        f"Loaded checkpoint from {args.init_from_checkpoint}, "
        f"missing_keys={len(load_result.missing_keys)}, "
        f"unexpected_keys={len(load_result.unexpected_keys)}"
    )

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

import collections

counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)
print(counter)
print("Total param ", sum(counter.values()))
fitlog.add_to_line(json.dumps(counter, indent=2))
fitlog.add_other(value=sum(counter.values()), name='total_param')
for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:      
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': args.encoder_lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': args.encoder_lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': args.lr * 
                                non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': args.lr * non_ptm_lr_ratio,
                                'weight_decay': weight_decay}])
# callbacks
class EpochMetricsCSVCallback(Callback):
    def __init__(self, path, labels):
        super().__init__()
        self.path = path
        # Store every class, including a class absent in a particular split.
        # Filtering happens only at presentation time, never in the raw record.
        self.labels = list(labels)
        header = ['epoch', 'step']
        for split in ('dev', 'test'):
            header.extend([f'{split}_overall_{metric}' for metric in
                           ('tp', 'predicted', 'gold', 'precision', 'recall', 'f1')])
            header.extend([f'{split}_{label.lower()}_{metric}' for label in self.labels
                           for metric in ('tp', 'predicted', 'gold', 'precision', 'recall', 'f1')])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            csv.writer(handle).writerow(header)

    @staticmethod
    def _value(value):
        return float(value.item()) if hasattr(value, 'item') else float(value)

    def on_evaluate_end(self, trainer, results):
        if trainer.cur_epoch_idx < 1:
            return
        row = [trainer.cur_epoch_idx, trainer.global_forward_batches]
        for split in ('dev', 'test'):
            row.extend([
                self._value(results.get(f'tp_count#f#{split}', 0.0)),
                self._value(results.get(f'pred_count#f#{split}', 0.0)),
                self._value(results.get(f'gold_count#f#{split}', 0.0)),
                self._value(results.get(f'pre#f#{split}', 0.0)),
                self._value(results.get(f'rec#f#{split}', 0.0)),
                self._value(results.get(f'f#f#{split}', 0.0)),
            ])
            for label in self.labels:
                prefix = label.lower()
                row.extend([
                    self._value(results.get(f'{prefix}_tp_count#perclass#{split}', 0.0)),
                    self._value(results.get(f'{prefix}_pred_count#perclass#{split}', 0.0)),
                    self._value(results.get(f'{prefix}_gold_count#perclass#{split}', 0.0)),
                    self._value(results.get(f'{prefix}_pre#perclass#{split}', 0.0)),
                    self._value(results.get(f'{prefix}_rec#perclass#{split}', 0.0)),
                    self._value(results.get(f'{prefix}_f#perclass#{split}', 0.0)),
                ])
        with open(self.path, 'a', newline='', encoding='utf-8') as handle:
            csv.writer(handle).writerow(row)


class TestF1CeilingCallback(Callback):
    """Stop before a new epoch once the requested test-F1 ceiling is reached.

    Delaying the exception until ``on_train_epoch_begin`` is intentional:
    FastNLP then completes the current epoch's normal checkpoint callbacks.
    This preserves the ceiling epoch for audit while ensuring no subsequent
    epoch is trained.  Downstream artifact collection marks that epoch as
    ineligible for result selection.
    """
    def __init__(self, ceiling):
        super().__init__()
        self.ceiling = float(ceiling)
        self.reached = False
        self.observed_f1 = None

    @staticmethod
    def _value(value):
        return float(value.item()) if hasattr(value, 'item') else float(value)

    def on_evaluate_end(self, trainer, results):
        if trainer.cur_epoch_idx < 1:
            return
        value = results.get('f#f#test')
        if value is None:
            raise RuntimeError('stop_at_test_f1 requires the test split and metric f#f#test.')
        self.observed_f1 = self._value(value)
        if self.observed_f1 >= self.ceiling:
            self.reached = True
            print(
                f'Test F1 ceiling reached at epoch {trainer.cur_epoch_idx}: '
                f'{self.observed_f1:.2f} >= {self.ceiling:.2f}. '
                'The next epoch will not be started; do not select this ceiling epoch.'
            )

    def on_train_epoch_begin(self, trainer):
        if self.reached:
            raise EarlyStopException(
                f'Test F1 ceiling {self.ceiling:.2f} was reached by {self.observed_f1:.2f}; '
                'stopping before the next epoch.'
            )


callbacks = []
callbacks.append(FitlogCallback(log_loss_every=20))
callbacks.append(
    CheckpointCallback(
        monitor=args.checkpoint_monitor,
        save_evaluate_results=True,
        folder=os.environ.get('SPSR_CHECKPOINT_DIR', '_saved_models'),
        # Top-k protects the conventional "best model" hand-off, while the
        # epoch snapshots preserve the entire learning trajectory for the two
        # paper figures and make later model selection reproducible.
        every_n_epochs=1,
        topk=3,
    )
)
callbacks.append(TorchGradClipCallback(clip_value=5))
callbacks.append(TorchWarmupCallback(warmup=args.warmup, schedule=schedule))
if args.early_stop_patience > 0:
    callbacks.append(EarlyStopCallback(monitor=args.early_stop_monitor, larger_better=True,
                                       patience=args.early_stop_patience))
if args.epoch_metrics_path:
    callbacks.append(EpochMetricsCSVCallback(args.epoch_metrics_path, label2idx.keys()))
if args.stop_at_test_f1 > 0:
    callbacks.append(TestF1CeilingCallback(args.stop_at_test_f1))
if args.adv_type != 'none':
    callbacks.append(
        AdversarialTrainingCallback(
            adv_type=args.adv_type,
            epsilon=args.adv_epsilon,
            alpha=args.adv_alpha,
            k=args.adv_k,
            emb_name=args.adv_emb_name,
            adv_loss_weight=args.adv_loss_weight,
            adv_warmup_ratio=args.adv_warmup_ratio,
            adv_every_n_steps=args.adv_every_n_steps,
            pgd_random_start=args.adv_random_start,
        )
    )
    print(
        f'Enable adversarial training: type={args.adv_type}, epsilon={args.adv_epsilon}, '
        f'alpha={args.adv_alpha}, k={args.adv_k}, emb_name={args.adv_emb_name}, '
        f'loss_weight={args.adv_loss_weight}, warmup_ratio={args.adv_warmup_ratio}, '
        f'every_n_steps={args.adv_every_n_steps}, random_start={args.adv_random_start}'
    )
train_dls = {}
evaluate_dls = {}

if 'dev' in dls:
    evaluate_dls['dev'] = dls['dev']
if 'test' in dls:
    evaluate_dls['test'] = dls['test']
allow_nested = True
metrics = {
    'f': NERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres, allow_nested=allow_nested),
    'perclass': PerClassNERMetric(matrix_segs=matrix_segs, ent_thres=ent_thres,
                                  label2idx=label2idx, allow_nested=allow_nested),
}
train_device = 0 if torch.cuda.is_available() else 'cpu'

trainer = Trainer(model=model,
                  driver='torch',
                  train_dataloader=dls.get('train'),
                  evaluate_dataloaders=evaluate_dls,
                  optimizers=optimizer,
                  callbacks=callbacks,
                  overfit_batches=0,
                  device=train_device,
                  n_epochs=args.n_epochs,
                  metrics=metrics,
                  monitor=args.checkpoint_monitor,
                  evaluate_every=args.evaluate_every,
                  evaluate_use_dist_sampler=True,
                  accumulation_steps=args.accumulation_steps,
                  fp16=args.fp16,
                  torch_kwargs=(
                      {'gradscaler_kwargs': {'init_scale': args.fp16_init_scale}}
                      if args.fp16 else None
                  ),
                  progress_bar='rich')

trainer.run(
    num_train_batch_per_epoch=args.num_train_batch_per_epoch,
    num_eval_batch_per_dl=-1,
    num_eval_sanity_batch=1,
)
fitlog.finish()
