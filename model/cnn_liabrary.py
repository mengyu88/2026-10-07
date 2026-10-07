import math
from torch import nn
import torch

import torch.nn.functional as F


class LocalSparseAttnSAD(nn.Module):
    """
    Local sparse-attention SAD path ported from the user's local-sparse-attn branch.

    This module builds local attention over the 8-neighbor window and outputs
    center-minus-context features.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        attn_dim=None,
        topk=4,
        use_rel_bias=True,
        use_gate=True,
    ):
        super(LocalSparseAttnSAD, self).__init__()
        if kernel_size != 3:
            raise ValueError("LocalSparseAttnSAD only supports kernel_size=3.")
        if groups != 1:
            raise ValueError("LocalSparseAttnSAD only supports groups=1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_dim = in_channels if attn_dim is None else attn_dim
        self.topk = topk
        self.use_rel_bias = use_rel_bias
        self.use_gate = use_gate

        unfold_padding = 1 if padding == 'same' else padding
        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=unfold_padding,
            stride=stride,
        )
        self.q_proj = nn.Conv2d(in_channels, self.attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, self.attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gate_proj = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=True)
        self.out_proj = (
            nn.Identity()
            if out_channels == in_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        )
        self.attn_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('neighbor_index', torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], dtype=torch.long))
        if use_rel_bias:
            self.rel_bias_scale = nn.Parameter(torch.tensor(0.1))
            self.rel_bias_mlp = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(16, 8, kernel_size=1, bias=True),
            )
        else:
            self.register_parameter('rel_bias_scale', None)
            self.rel_bias_mlp = None

    @staticmethod
    def _build_relation_feat(h, w, device, dtype):
        row_idx = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
        col_idx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
        span_len = col_idx - row_idx
        max_span = float(max(h - 1, 1))
        span_norm = span_len / max_span
        near_diag = (span_len.abs() <= 1).to(dtype)
        return torch.cat([span_norm, near_diag], dim=1)

    def forward(self, x):
        bsz, c, h, w = x.shape
        kernel_size = 3
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k_unfold = self.unfold(k).reshape(bsz, self.attn_dim, kernel_size * kernel_size, h, w)
        v_unfold = self.unfold(v).reshape(bsz, c, kernel_size * kernel_size, h, w)

        logits = (q.unsqueeze(2) * k_unfold).sum(dim=1) / math.sqrt(float(self.attn_dim))
        logits = logits[:, self.neighbor_index, :, :]

        valid_mask = x.new_ones((bsz, 1, h, w))
        valid_neighbors = self.unfold(valid_mask).reshape(bsz, 1, 9, h, w)[:, :, self.neighbor_index, :, :]
        valid_neighbors = valid_neighbors.squeeze(1) > 0.5
        logits = logits.masked_fill(~valid_neighbors, -1e4)

        if self.use_rel_bias:
            rel_feat = self._build_relation_feat(h, w, x.device, logits.dtype)
            rel_bias = self.rel_bias_mlp(rel_feat)
            logits = logits + self.rel_bias_scale * rel_bias

        k_top = min(max(1, int(self.topk)), logits.size(1))
        sparse = soft_topk_mask(logits, topk=k_top, temperature=1.0)
        dense = torch.softmax(logits, dim=1)
        mix = torch.sigmoid(self.attn_mix_logit)
        attn = mix * sparse + (1.0 - mix) * dense
        attn = attn * valid_neighbors.to(attn.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        center = v_unfold[:, :, 4, :, :]
        value_nb = v_unfold[:, :, self.neighbor_index, :, :]
        context = (value_nb * attn.unsqueeze(1)).sum(dim=2)
        delta = center - context

        if self.use_gate:
            gate = torch.sigmoid(self.gate_proj(torch.cat([center, context], dim=1)))
            delta = gate * delta
        return self.out_proj(delta)


def soft_topk_mask(logits, topk=2, temperature=1.0):
    temperature = max(float(temperature), 1e-6)
    scaled = logits / temperature
    if topk is None or topk <= 0 or topk >= scaled.size(1):
        return F.softmax(scaled, dim=1)
    topk_values, topk_indices = torch.topk(scaled, k=topk, dim=1)
    topk_probs = F.softmax(topk_values, dim=1)
    output = torch.zeros_like(scaled)
    output.scatter_(1, topk_indices, topk_probs)
    return output


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x
