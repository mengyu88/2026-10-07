from torch import nn
import torch
import torch.nn.functional as F


def _resize_mask(mask, x):
    if mask.size(-1) == x.size(-1) and mask.size(-2) == x.size(-2):
        return mask
    return F.interpolate(mask.float(), size=x.shape[-2:], mode='nearest').bool()


class _LayerNorm2d(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(_LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class _BFMConvBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super(_BFMConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.norm = _LayerNorm2d((1, channels, 1, 1), dim_index=1)
        self.act = nn.GELU()

    def forward(self, x, mask):
        mask = _resize_mask(mask, x)
        y = x.masked_fill(mask, 0)
        y = self.conv(y)
        y = self.norm(y)
        y = self.act(y)
        return y


class PyramidGatedBFM(nn.Module):
    """
    A detail-preserving BFM: one down-sampling stage + gated fusion.
    """

    def __init__(self, channels):
        super(PyramidGatedBFM, self).__init__()
        self.enc0 = _BFMConvBlock(channels, stride=1)
        self.enc1 = _BFMConvBlock(channels, stride=2)
        self.bridge = _BFMConvBlock(channels, stride=1)
        self.dec0 = _BFMConvBlock(channels, stride=1)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, mask):
        m0 = _resize_mask(mask, x)
        e0 = self.enc0(x, m0)
        e1 = self.enc1(e0, _resize_mask(m0, e0))
        b = self.bridge(e1, _resize_mask(m0, e1))

        u0 = F.interpolate(b, size=e0.shape[-2:], mode='nearest') + e0
        u0 = self.dec0(u0, _resize_mask(m0, u0))

        gate_in = torch.cat([u0, e0], dim=1).masked_fill(m0, 0)
        gate = torch.sigmoid(self.gate(gate_in))
        fused = gate * u0 + (1.0 - gate) * e0
        out = self.out(fused.masked_fill(m0, 0))
        return out.masked_fill(m0, 0)
