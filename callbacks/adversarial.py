from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from fastNLP.core.callbacks import Callback


class AdversarialTrainingCallback(Callback):
    def __init__(
        self,
        adv_type: str = 'pgd',
        epsilon: float = 1.0,
        alpha: float = 0.3,
        k: int = 3,
        emb_name: str = 'word_embeddings',
        adv_loss_weight: float = 1.0,
        adv_warmup_ratio: float = 0.0,
        adv_every_n_steps: int = 1,
        pgd_random_start: bool = False,
    ):
        super().__init__()
        if adv_type != 'pgd':
            raise ValueError(f"adv_type must be 'pgd', got {adv_type}")
        if epsilon <= 0:
            raise ValueError(f'epsilon must be > 0, got {epsilon}')
        if alpha <= 0:
            raise ValueError(f'alpha must be > 0, got {alpha}')
        if k <= 0:
            raise ValueError(f'k must be > 0, got {k}')
        if adv_loss_weight < 0:
            raise ValueError(f'adv_loss_weight must be >= 0, got {adv_loss_weight}')
        if not (0 <= adv_warmup_ratio <= 1):
            raise ValueError(f'adv_warmup_ratio must be in [0, 1], got {adv_warmup_ratio}')
        if adv_every_n_steps <= 0:
            raise ValueError(f'adv_every_n_steps must be > 0, got {adv_every_n_steps}')
        self.adv_type = adv_type
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.k = int(k)
        self.emb_name = emb_name
        self.adv_loss_weight = float(adv_loss_weight)
        self.adv_warmup_ratio = float(adv_warmup_ratio)
        self.adv_every_n_steps = int(adv_every_n_steps)
        self.pgd_random_start = bool(pgd_random_start)
        self._batch = None
        self._emb_backup: Dict[str, torch.Tensor] = {}
        self._grad_backup: Dict[str, torch.Tensor] = {}

    def on_train_batch_begin(self, trainer, batch, indices):
        self._batch = batch

    def on_after_backward(self, trainer):
        if self._batch is None:
            return
        if not self._should_attack(trainer):
            return
        adv_scale = self._get_adv_loss_scale(trainer)
        if adv_scale <= 0:
            return
        model = trainer.driver.unwrap_model()
        self._run_pgd(trainer, model, adv_scale)

    def _iter_embedding_params(self, model) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.emb_name not in name:
                continue
            if param.grad is None:
                continue
            yield name, param

    @staticmethod
    def _safe_l2_norm(t: torch.Tensor) -> torch.Tensor:
        return torch.norm(t)

    def _backup_all_grads(self, model):
        self._grad_backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self._grad_backup[name] = param.grad.detach().clone()

    def _restore_all_grads(self, model):
        for name, param in model.named_parameters():
            if name not in self._grad_backup:
                continue
            if param.grad is None:
                param.grad = self._grad_backup[name].detach().clone()
            else:
                param.grad.data.copy_(self._grad_backup[name].data)

    def _forward_backward_once(self, trainer, adv_scale: float):
        adv_outputs = trainer.train_step(self._batch)
        adv_loss = trainer.extract_loss_from_outputs(adv_outputs)
        adv_loss = adv_loss * adv_scale / trainer.accumulation_steps
        trainer.driver.backward(adv_loss)

    def _project(self, param_name: str, param: torch.nn.Parameter):
        delta = param.data - self._emb_backup[param_name]
        delta_norm = self._safe_l2_norm(delta)
        if torch.isfinite(delta_norm) and delta_norm > self.epsilon:
            param.data.copy_(self._emb_backup[param_name] + self.epsilon * delta / (delta_norm + 1e-12))

    def _run_pgd(self, trainer, model, adv_scale: float):
        params = list(self._iter_embedding_params(model))
        if len(params) == 0:
            return

        self._emb_backup = {name: param.data.detach().clone() for name, param in params}
        self._backup_all_grads(model)
        if self.pgd_random_start:
            for name, param in params:
                delta = torch.empty_like(param).uniform_(-self.epsilon, self.epsilon)
                param.data.add_(delta)
                self._project(name, param)

        for t in range(self.k):
            for name, param in params:
                grad = param.grad
                if grad is None:
                    continue
                norm = self._safe_l2_norm(grad)
                if torch.isfinite(norm) and norm > 0:
                    r_at = self.alpha * grad / (norm + 1e-12)
                    param.data.add_(r_at)
                    self._project(name, param)

            if t != self.k - 1:
                trainer.driver.zero_grad()
            else:
                self._restore_all_grads(model)

            self._forward_backward_once(trainer, adv_scale)

        for name, param in params:
            param.data.copy_(self._emb_backup[name])

        self._emb_backup = {}
        self._grad_backup = {}

    def _should_attack(self, trainer) -> bool:
        if self.adv_every_n_steps == 1:
            return True
        step = getattr(trainer, 'global_forward_batches', 0) + 1
        return step % self.adv_every_n_steps == 0

    def _get_adv_loss_scale(self, trainer) -> float:
        if self.adv_loss_weight <= 0:
            return 0.0
        if self.adv_warmup_ratio <= 0:
            return self.adv_loss_weight
        total_steps = getattr(trainer, 'n_batches', 0)
        if total_steps is None or total_steps <= 0:
            return self.adv_loss_weight
        warmup_steps = max(1, int(total_steps * self.adv_warmup_ratio))
        cur_step = getattr(trainer, 'global_forward_batches', 0) + 1
        warmup_scale = min(1.0, cur_step / float(warmup_steps))
        return self.adv_loss_weight * warmup_scale
