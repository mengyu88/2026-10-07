from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask

try:
    from torch_scatter import scatter_max
except ImportError:
    # The project only needs max pooling over subword positions. PyTorch 2.x
    # provides the equivalent operation when torch-scatter is unavailable.
    def scatter_max(src, index, dim=1):
        if dim != 1:
            raise NotImplementedError('The torch_scatter fallback only supports dim=1')
        expanded_index = index.unsqueeze(-1).expand_as(src)
        output_size = int(index.max().item()) + 1
        output = src.new_zeros((src.size(0), output_size, src.size(2)))
        output.scatter_reduce_(1, expanded_index, src, reduce='amax', include_self=False)
        return output, None

import torch
import torch.nn.functional as F

from .cnn import MaskCNN_1, MaskCNN_2
from .multi_head_biaffine3 import MultiHeadBiaffine


class CNNNer(nn.Module):
    """SPSR-Net: biaffine span features + SNSA + HSR + a linear ASL classifier."""

    def __init__(
        self,
        model_name,
        num_ner_tag,
        cnn_dim=200,
        biaffine_size=200,
        size_embed_dim=0,
        logit_drop=0,
        kernel_size=3,
        n_head=4,
        cnn_depth=3,
        n_layer=2,
        separateness_rate=0.1,
        theta=1,
        loss_theta=1,
        sad_topk=2,
        sad_attn_dim=None,
        use_snsa=True,
        use_hsr=True,
        sad_use_rel_bias=True,
        sad_gate=True,
        loss_type='bce',
        asl_gamma_pos=0.0,
        asl_gamma_neg=3.0,
        asl_clip=0.05,
        subword_pooling='max',
    ):
        super().__init__()
        if cnn_depth <= 0:
            raise ValueError('cnn_depth must be > 0 for the SPSR feature extractor')
        if n_layer not in (1, 2):
            raise ValueError('n_layer must be 1 or 2')
        if loss_type not in ('bce', 'asl'):
            raise ValueError("loss_type must be 'bce' or 'asl'")
        if subword_pooling not in ('max', 'legacy_zero_clamped'):
            raise ValueError('Unsupported subword pooling mode')

        self.mdim = cnn_dim
        self.num_ner_tag = num_ner_tag
        self.cnn_dim = cnn_dim
        self.separateness_rate = separateness_rate
        self.loss_theta = loss_theta
        self.n_layer = n_layer
        self.loss_type = loss_type
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_clip = asl_clip
        self.subword_pooling = subword_pooling

        self.pretrain_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.pretrain_model.config.hidden_size
        if size_embed_dim != 0:
            n_pos = 30
            self.size_embedding = nn.Embedding(n_pos, size_embed_dim)
            span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            span_size_ids.masked_fill_(span_size_ids < -n_pos / 2, -n_pos / 2)
            span_size_ids = span_size_ids.masked_fill(span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', span_size_ids.long())
            hsz = biaffine_size * 2 + size_embed_dim + 2
        else:
            hsz = biaffine_size * 2 + 2

        self.dropout = nn.Dropout(logit_drop)
        self.dropout1 = nn.Dropout(logit_drop)
        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, biaffine_size),
            nn.GELU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, biaffine_size),
            nn.GELU(),
        )
        if n_head > 0:
            self.biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            nn.init.xavier_normal_(self.U.data)
        self.W = nn.Parameter(torch.empty(cnn_dim, hsz))
        nn.init.xavier_normal_(self.W.data)

        cnn_class = MaskCNN_1 if n_layer == 1 else MaskCNN_2
        self.cnn1 = cnn_class(
            cnn_dim,
            cnn_dim,
            kernel_size=kernel_size,
            depth=cnn_depth,
            theta=theta,
            sad_topk=sad_topk,
            sad_attn_dim=sad_attn_dim,
            use_snsa=use_snsa,
            use_hsr=use_hsr,
            sad_use_rel_bias=sad_use_rel_bias,
            sad_gate=sad_gate,
        )
        self.score_head = nn.Linear(cnn_dim * 3, num_ner_tag)
        nn.init.xavier_normal_(self.score_head.weight.data)
        self.logit_drop = logit_drop

    def _asl_loss_matrix(self, logits, targets, valid_mask):
        prob = torch.sigmoid(logits)
        positive = targets.gt(0.5) & valid_mask
        negative = targets.le(0.5) & valid_mask
        loss = logits.new_zeros(logits.shape)

        neg_prob = (1.0 - prob).clamp(min=1e-6, max=1.0 - 1e-6)
        if self.asl_clip > 0:
            neg_prob = (neg_prob + self.asl_clip).clamp(min=1e-6, max=1.0 - 1e-6)
        neg_loss = -torch.log(neg_prob)
        if self.asl_gamma_neg > 0:
            neg_loss = neg_loss * torch.pow((1.0 - neg_prob).clamp(min=1e-6), self.asl_gamma_neg)
        loss = torch.where(negative, neg_loss, loss)

        pos_prob = prob.clamp(min=1e-6, max=1.0 - 1e-6)
        pos_loss = -torch.log(pos_prob)
        if self.asl_gamma_pos > 0:
            pos_loss = pos_loss * torch.pow((1.0 - pos_prob).clamp(min=1e-6), self.asl_gamma_pos)
        return torch.where(positive, pos_loss, loss)

    def _span_loss(self, final_score, matrix, batch_size):
        valid_mask = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_mask, 0.0).float()
        if self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(final_score, targets, reduction='none')
        else:
            loss = self._asl_loss_matrix(final_score.float(), targets, valid_mask)
        sample_mask = valid_mask.float().view(batch_size, -1)
        return ((loss.view(batch_size, -1) * sample_mask).sum(dim=-1)).mean()

    def forward(self, input_ids, bpe_len, indexes, matrix, raw_words):
        attention_mask = seq_len_to_mask(bpe_len)
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        scat_max = scatter_max(last_hidden_states, index=indexes, dim=1)[0]
        if self.subword_pooling == 'legacy_zero_clamped':
            scat_max = scat_max.clamp_min(0)
        state = scat_max[:, 1:]
        lengths, _ = indexes.max(dim=-1)
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.biaffine(head_state, tail_state)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat(
            [
                head_state.unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                tail_state.unsqueeze(1).expand(-1, head_state.size(1), -1, -1),
            ],
            dim=-1,
        )
        mask = seq_len_to_mask(lengths)
        mask = mask[:, None] * mask.unsqueeze(-1)
        pad_mask = mask[:, None].eq(0)
        pad_mask1 = pad_mask * torch.tril(pad_mask).ne(0)
        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat(
                [
                    self.dropout(affined_cat),
                    self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1),
                ],
                dim=-1,
            )

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)
        scores = scores2 + scores1
        if self.logit_drop != 0:
            scores = F.dropout(scores, p=self.logit_drop, training=self.training)
        spsr_features = self.cnn1(scores.masked_fill(pad_mask1, 0), pad_mask1, self.training)
        span_features = torch.concat([scores, spsr_features], dim=1).permute(0, 2, 3, 1)
        final_score = self.score_head(span_features)
        assert final_score.size(-1) == matrix.size(-1)
        if self.training:
            return {'loss': self._span_loss(final_score, matrix, input_ids.size(0))}
        return {'scores': final_score}
