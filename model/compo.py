'''
@author: Y.J. Lee
'''

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class PatchEmbedding(nn.Module):
    """Converts patch token indices into patch embeddings.

    Input:  (batch, num_patches, w) LongTensor of token indices
    Output: (batch, num_patches, d_model)
    """

    def __init__(self, vocab_size: int, d_emb: int, d_model: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.d_emb = d_emb

        # Token embedding with padding_idx=0 (zero vector for padding)
        self.token_emb = nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)
        # Learnable within-patch positional encoding for positions 0..w-1
        self.pos_emb = nn.Embedding(window_size, d_emb)
        # RMSNorm applied on last dimension
        self.norm = RMSNorm(d_emb)
        # Projection from flattened (w * d_emb) to d_model
        self.proj = nn.Linear(window_size * d_emb, d_model)

    def forward(self, x):
        # x: (batch, num_patches, w)
        batch, num_patches, w = x.shape

        # Token embedding: (batch, num_patches, w, d_emb)
        tok = self.token_emb(x)

        # Positional embedding: (w,) -> (w, d_emb), broadcast to match
        positions = torch.arange(w, device=x.device)
        pos = self.pos_emb(positions)  # (w, d_emb)

        # Add token + positional embeddings
        out = tok + pos  # (batch, num_patches, w, d_emb)

        # RMSNorm on last dimension
        out = self.norm(out)  # (batch, num_patches, w, d_emb)

        # Flatten last two dims and project
        out = out.reshape(batch, num_patches, w * self.d_emb)  # (batch, num_patches, w * d_emb)
        out = self.proj(out)  # (batch, num_patches, d_model)

        return out


class ChannelEncoder(nn.Module):
    """Standard Transformer encoder for one channel.

    Input:  (batch, num_patches, d_model) patch embeddings
    Output: (batch, num_patches, d_model) encoded representations
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, d_ff: int, max_patches: int):
        super().__init__()
        # Learnable sequence-level positional embedding
        self.pos_emb = nn.Embedding(max_patches, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, num_patches, d_model)
        num_patches = x.size(1)

        # Add sequence-level positional embedding
        positions = torch.arange(num_patches, device=x.device)
        x = x + self.pos_emb(positions)  # (batch, num_patches, d_model)

        # Transformer encoder
        out = self.encoder(x)  # (batch, num_patches, d_model)

        return out


class ChannelDecoderAct(nn.Module):
    """Activity channel decoder.

    Input:  patch_emb      (batch, num_patches, d_model) — decoder input patch embeddings
            enc_output     (batch, num_patches, d_model) — encoder output C_act
            teacher_tokens (batch, num_patches) — teacher forcing token indices
    Output: logits (batch, num_patches, w, vocab_size+1) — predictions for each position in patch
            hc_act (batch, num_patches, d_model) — cross-attention outputs
    """

    def __init__(self, vocab_size: int, d_emb: int, d_model: int, nhead: int, d_gru: int, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.d_gru = d_gru
        self.window_size = window_size
        self.vocab_size = vocab_size

        # Cross-attention: Q=patch_emb, K/V=enc_output
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Teacher token embedding
        self.teacher_emb = nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)

        # GRU cell: input is cat[hc_act_t, teacher_emb_t]
        self.gru_cell = nn.GRUCell(d_model + d_emb, d_gru)

        # Output head: project from d_gru to window_size * (vocab_size + 1)
        self.output_head = nn.Linear(d_gru, window_size * (vocab_size + 1))

    def forward(self, patch_emb, enc_output, teacher_tokens):
        # patch_emb:      (batch, num_patches, d_model)
        # enc_output:     (batch, num_patches, d_model)
        # teacher_tokens: (batch, num_patches)
        batch, num_patches, _ = patch_emb.shape

        # Cross-attention over all steps at once
        hc_act, _ = self.cross_attn(patch_emb, enc_output, enc_output)
        # hc_act: (batch, num_patches, d_model)

        # Teacher token embedding
        t_emb = self.teacher_emb(teacher_tokens)  # (batch, num_patches, d_emb)

        # Sequential GRU loop over num_patches steps
        h = torch.zeros(batch, self.d_gru, device=patch_emb.device)  # initial hidden state
        all_logits = []

        for t in range(num_patches):
            # Concatenate cross-attention output and teacher embedding at step t
            gru_input = torch.cat([hc_act[:, t, :], t_emb[:, t, :]], dim=-1)  # (batch, d_model + d_emb)
            h = self.gru_cell(gru_input, h)  # (batch, d_gru)

            # Output head
            out = self.output_head(h)  # (batch, window_size * (vocab_size + 1))
            out = out.reshape(batch, self.window_size, self.vocab_size + 1)  # (batch, w, vocab_size+1)
            all_logits.append(out)

        # Stack all step outputs: (batch, num_patches, w, vocab_size+1)
        logits = torch.stack(all_logits, dim=1)

        return logits, hc_act


class ChannelDecoderAttr(nn.Module):
    """Attribute channel decoder.

    Same structure as ChannelDecoderAct but GRU input includes both the activity
    cross-attention output (hc_act) and this channel's own cross-attention output.

    Input:  patch_emb      (batch, num_patches, d_model) — this channel's decoder input embeddings
            enc_output     (batch, num_patches, d_model) — this channel's encoder output
            teacher_tokens (batch, num_patches) — teacher forcing tokens
            hc_act         (batch, num_patches, d_model) — activity cross-attention outputs from DecoderAct
    Output: logits (batch, num_patches, w, vocab_size+1)
    """

    def __init__(self, vocab_size: int, d_emb: int, d_model: int, nhead: int, d_gru: int, window_size: int):
        super().__init__()
        self.d_model = d_model
        self.d_emb = d_emb
        self.d_gru = d_gru
        self.window_size = window_size
        self.vocab_size = vocab_size

        # Cross-attention: Q=patch_emb, K/V=enc_output
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Teacher token embedding
        self.teacher_emb = nn.Embedding(vocab_size + 1, d_emb, padding_idx=0)

        # GRU cell: input is cat[hc_act_t, hc_attr_t, teacher_emb_t]
        self.gru_cell = nn.GRUCell(2 * d_model + d_emb, d_gru)

        # Output head
        self.output_head = nn.Linear(d_gru, window_size * (vocab_size + 1))

    def forward(self, patch_emb, enc_output, teacher_tokens, hc_act):
        # patch_emb:      (batch, num_patches, d_model)
        # enc_output:     (batch, num_patches, d_model)
        # teacher_tokens: (batch, num_patches)
        # hc_act:         (batch, num_patches, d_model)
        batch, num_patches, _ = patch_emb.shape

        # Cross-attention over all steps at once
        hc_attr, _ = self.cross_attn(patch_emb, enc_output, enc_output)
        # hc_attr: (batch, num_patches, d_model)

        # Teacher token embedding
        t_emb = self.teacher_emb(teacher_tokens)  # (batch, num_patches, d_emb)

        # Sequential GRU loop over num_patches steps
        h = torch.zeros(batch, self.d_gru, device=patch_emb.device)  # initial hidden state
        all_logits = []

        for t in range(num_patches):
            # Concatenate activity cross-attn, this channel's cross-attn, and teacher embedding
            gru_input = torch.cat([hc_act[:, t, :], hc_attr[:, t, :], t_emb[:, t, :]], dim=-1)
            # (batch, 2 * d_model + d_emb)
            h = self.gru_cell(gru_input, h)  # (batch, d_gru)

            # Output head
            out = self.output_head(h)  # (batch, window_size * (vocab_size + 1))
            out = out.reshape(batch, self.window_size, self.vocab_size + 1)  # (batch, w, vocab_size+1)
            all_logits.append(out)

        # Stack all step outputs: (batch, num_patches, w, vocab_size+1)
        logits = torch.stack(all_logits, dim=1)

        return logits
