'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with configurable input/output dimensions.

    Reference: Shazeer, "GLU Variants Improve Transformer" (arXiv:2002.05202)
    """
    def __init__(self, d_in: int, d_out: int, d_ffn: int = None):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_out
        hidden_dim = int(2 * d_ffn / 3)
        self.w1 = nn.Linear(d_in, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_in, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_out, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class PatchEmbedding(nn.Module):
    """(batch, num_patches, w) LongTensor -> (batch, num_patches, d_model)

    Within-patch position is implicitly captured by flatten+linear.
    Patch-level position is added via learnable embeddings.
    """

    def __init__(self, vocab_size: int, d_emb: int, d_model: int,
                 window_size: int, max_patches: int):
        super().__init__()
        self.d_emb = d_emb
        self.token_emb = nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)
        self.norm = RMSNorm(d_emb)
        self.proj = nn.Linear(window_size * d_emb, d_model, bias=False)
        self.pos_emb = nn.Embedding(max_patches, d_model)

    def forward(self, x):
        B, P, w = x.shape
        tok = F.dropout(self.token_emb(x), p=0.1, training=self.training)
        tok = self.norm(tok)
        out = self.proj(tok.reshape(B, P, w * self.d_emb))
        pos = self.pos_emb(torch.arange(P, device=x.device))
        return out + pos


class ChannelEncoder(nn.Module):
    """(batch, P, d_model) -> (batch, P, d_model) via TransformerEncoder."""

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        return self.encoder(x, src_key_padding_mask=padding_mask)


def _masked_mean_pool(x, mask):
    """Mean-pool over dim=1, excluding positions where mask is True."""
    if mask is not None:
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return x.mean(dim=1)


class ChannelDecoderAct(nn.Module):
    """Activity channel decoder.

    Cross-attention K/V comes from SwiGLU-fused encoder outputs.
    GRU h0 is initialized from own channel's encoder output.
    Returns logits and detached hc_act for attribute decoders.
    """

    def __init__(self, vocab_size: int, d_model: int, d_gru: int,
                 window_size: int, num_dec_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_gru = d_gru
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.num_dec_layers = num_dec_layers
        self.p = dropout

        self.cross_attn = nn.MultiheadAttention(
            d_model, 1, dropout=0.1, batch_first=True)
        self.norm = RMSNorm(d_model)
        self.gru = nn.GRU(d_model, d_gru, num_layers=num_dec_layers,
                          batch_first=True, dropout=dropout)
        self.head = nn.Linear(
            d_model + d_gru, window_size * (vocab_size + 1), bias=False)

    def forward(self, patch_emb, enc_output, enc_fused,
                padding_mask=None, cross_padding_mask=None):
        B, P, _ = patch_emb.shape

        hc, _ = self.cross_attn(patch_emb, enc_fused, enc_fused,
                                key_padding_mask=cross_padding_mask)
        hc = self.norm(hc + patch_emb)
        context_act = hc.detach().clone()

        h0 = _masked_mean_pool(enc_output, padding_mask)
        h0 = h0.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
        gru_out, _ = self.gru(hc, h0)

        out = torch.cat([
            F.dropout(patch_emb, p=self.p, training=self.training),
            gru_out], dim=-1)
        logits = self.head(out).reshape(B, P, self.window_size, self.vocab_size + 1)

        return logits, context_act


class ChannelDecoderAttr(nn.Module):
    """Attribute channel decoder.

    Cross-attention K/V comes from SwiGLU-fused encoder outputs.
    GRU input includes both hc_attr and hc_act (from activity decoder, detached).
    Output head receives patch_emb, hc_act, and gru_out.
    """

    def __init__(self, vocab_size: int, d_model: int, d_gru: int,
                 window_size: int, num_dec_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.d_gru = d_gru
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.num_dec_layers = num_dec_layers
        self.p = dropout

        self.cross_attn = nn.MultiheadAttention(
            d_model, 1, dropout=0.1, batch_first=True)
        self.norm = RMSNorm(d_model)
        self.gru = nn.GRU(2 * d_model, d_gru, num_layers=num_dec_layers,
                          batch_first=True, dropout=dropout)
        self.head = nn.Linear(
            d_model + d_model + d_gru, window_size * (vocab_size + 1), bias=False)

    def forward(self, patch_emb, enc_output, enc_fused, hc_act,
                padding_mask=None, cross_padding_mask=None):
        B, P, _ = patch_emb.shape

        hc, _ = self.cross_attn(patch_emb, enc_fused, enc_fused,
                                key_padding_mask=cross_padding_mask)
        hc = self.norm(hc + patch_emb)

        h0 = _masked_mean_pool(enc_output, padding_mask)
        h0 = h0.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
        gru_out, _ = self.gru(torch.cat([hc, hc_act], dim=-1), h0)

        out = torch.cat([
            F.dropout(patch_emb, p=self.p, training=self.training),
            F.dropout(hc_act, p=self.p, training=self.training),
            gru_out,
        ], dim=-1)
        logits = self.head(out).reshape(B, P, self.window_size, self.vocab_size + 1)

        return logits
