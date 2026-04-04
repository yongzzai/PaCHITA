'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class PatchEmbedding(nn.Module):
    """Converts patch token indices into patch embeddings.

    Input:  (batch, num_patches, w) LongTensor of token indices
    Output: (batch, num_patches, d_model)

    Within-patch positional information is implicitly captured by flatten+linear
    (each token position occupies a fixed slot in the flattened vector).
    Patch-level positional encoding is added via learnable embeddings.
    """

    def __init__(self, vocab_size: int, d_emb: int, d_model: int, window_size: int, max_patches: int):
        super().__init__()
        self.window_size = window_size
        self.d_emb = d_emb

        self.token_emb = nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)
        self.norm = RMSNorm(d_emb)
        self.proj = nn.Linear(window_size * d_emb, d_model, bias=False)
        self.patch_pos_emb = nn.Embedding(max_patches, d_model)

    def forward(self, x):
        # x: (batch, num_patches, w)
        batch, num_patches, w = x.shape
        tok = self.token_emb(x)  # (batch, num_patches, w, d_emb)
        out = self.norm(tok)  # (batch, num_patches, w, d_emb)
        out = self.proj(out.reshape(batch, num_patches, w * self.d_emb))  # (batch, num_patches, d_model)
        positions = torch.arange(num_patches, device=x.device)
        return out + self.patch_pos_emb(positions)


class ChannelEncoder(nn.Module):
    """Standard Transformer encoder for one channel.

    Input:  (batch, num_patches, d_model) patch embeddings
    Output: (batch, num_patches, d_model) encoded representations
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        return self.encoder(x, src_key_padding_mask=padding_mask)


class ChannelDecoderAct(nn.Module):
    """Activity channel decoder.

    Input:  patch_emb      (batch, num_patches, d_model) — decoder input patch embeddings
            enc_output     (batch, num_patches, d_model) — encoder output C_act
    Output: logits (batch, num_patches, w, vocab_size+1) — predictions for each position in patch
            hc_act (batch, num_patches, d_model) — cross-attention outputs (detached)
    """

    def __init__(self, vocab_size: int, d_model: int,
                 d_gru: int, window_size: int, num_dec_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_gru = d_gru
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.p = dropout

        # Cross-attention: Q=patch_emb, K/V=enc_output
        self.cross_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1, batch_first=True)
        self.norm = RMSNorm(d_model)

        # Projection: enc_output mean-pool → GRU initial hidden state
        self.h0_proj = nn.Linear(d_model, d_gru, bias=False)
        self.num_dec_layers = num_dec_layers

        self.gru = nn.GRU(d_model, d_gru, num_layers=num_dec_layers, batch_first=True, bias=False, dropout=self.p)

        # Output head: input is cat[patch_emb, gru_out]
        self.output_head = nn.Linear(d_model + d_gru, window_size * (vocab_size + 1), bias=False)

    def forward(self, patch_emb, enc_output, padding_mask=None):
        # patch_emb:      (batch, num_patches, d_model)
        # enc_output:     (batch, num_patches, d_model)
        # padding_mask:   (batch, num_patches) True where patch is entirely padding
        batch, num_patches, _ = patch_emb.shape

        # Causal mask: prevent attending to future encoder patches
        causal_mask = torch.triu(torch.ones(num_patches, num_patches, device=patch_emb.device, dtype=torch.bool), diagonal=1)

        # Cross-attention over all steps at once
        hc_act, _ = self.cross_attn(patch_emb, enc_output, enc_output,
                                    attn_mask=causal_mask,
                                    key_padding_mask=padding_mask)
        hc_act = self.norm(hc_act + patch_emb)  # residual + norm
        context_act = hc_act.detach().clone()

        # GRU initial state from encoder output (masked mean-pool)
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()  # (batch, num_patches, 1)
            h0 = self.h0_proj((enc_output * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1))
        else:
            h0 = self.h0_proj(enc_output.mean(dim=1))
        h0 = h0.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()  # (num_layers, batch, d_gru)

        # GRU over all steps: input is hc_act only
        gru_output, _ = self.gru(hc_act, h0)  # (batch, num_patches, d_gru)

        head_input = torch.cat([F.dropout(patch_emb, p=self.p, training=self.training), gru_output], dim=-1)  # (batch, num_patches, d_model + d_gru)
        logits = self.output_head(head_input)  # (batch, num_patches, w * (vocab_size + 1))
        logits = logits.reshape(batch, num_patches, self.window_size, self.vocab_size + 1)

        return logits, context_act


class ChannelDecoderAttr(nn.Module):
    """Attribute channel decoder.

    Input:  patch_emb      (batch, num_patches, d_model) — this channel's decoder input embeddings
            enc_output     (batch, num_patches, d_model) — this channel's encoder output
            hc_act         (batch, num_patches, d_model) — activity cross-attention outputs (detached)
    Output: logits (batch, num_patches, w, vocab_size+1)
    """

    def __init__(self, vocab_size: int, d_model: int,
                 d_gru: int, window_size: int, num_dec_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_gru = d_gru
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.p = dropout

        self.cross_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1, batch_first=True)
        self.norm = RMSNorm(d_model)

        self.h0_proj = nn.Linear(d_model, d_gru, bias=False)
        self.num_dec_layers = num_dec_layers

        # GRU: input is cat[hc_attr, hc_act]
        self.gru = nn.GRU(2 * d_model, d_gru, num_layers=num_dec_layers, batch_first=True, bias=False, dropout=self.p)
        
        # Output head: input is cat[gru_out, patch_emb, hc_act]
        self.output_head = nn.Linear(d_gru + d_model + d_model, window_size * (vocab_size + 1), bias=False)

    def forward(self, patch_emb, enc_output, hc_act, padding_mask=None):
        # patch_emb:      (batch, num_patches, d_model)
        # enc_output:     (batch, num_patches, d_model)
        # hc_act:         (batch, num_patches, d_model)
        # padding_mask:   (batch, num_patches) True where patch is entirely padding
        batch, num_patches, _ = patch_emb.shape

        # Causal mask: prevent attending to future encoder patches
        causal_mask = torch.triu(torch.ones(num_patches, num_patches, device=patch_emb.device, dtype=torch.bool), diagonal=1)

        # Cross-attention over all steps at once
        hc_attr, _ = self.cross_attn(patch_emb, enc_output, enc_output,
                                     attn_mask=causal_mask,
                                     key_padding_mask=padding_mask)
        hc_attr = self.norm(hc_attr + patch_emb)  # residual + norm

        # GRU initial state from encoder output (masked mean-pool)
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()  # (batch, num_patches, 1)
            h0 = self.h0_proj((enc_output * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1))
        else:
            h0 = self.h0_proj(enc_output.mean(dim=1))
        h0 = h0.unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()  # (num_layers, batch, d_gru)

        # GRU over all steps: input is cat[hc_attr, hc_act]
        gru_input = torch.cat([hc_attr, hc_act], dim=-1)  # (batch, num_patches, 2*d_model)
        gru_output, _ = self.gru(gru_input, h0)  # (batch, num_patches, d_gru)

        head_input = torch.cat([F.dropout(patch_emb, p=self.p, training=self.training), hc_act, gru_output], dim=-1)
        logits = self.output_head(head_input)  # (batch, num_patches, w * (vocab_size + 1))
        logits = logits.reshape(batch, num_patches, self.window_size, self.vocab_size + 1)

        return logits
