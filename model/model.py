'''
PaCHITA: Patch-based Business Process Anomaly Detection

@author: Y.J. Lee
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.compo import (
    PatchEmbedding, ChannelEncoder, ChannelDecoderAct, ChannelDecoderAttr, SwiGLUFFN,
)


class PaCHITA:
    """Wrapper class with fit/detect interface for PaCHITA anomaly detection."""

    name = 'PaCHITA'

    def __init__(self, window_size=3, d_emb=16, d_model=64, nhead=4,
                 num_enc_layers=2, num_dec_layers=2, d_ff=128, d_gru=64,
                 enc_dropout=0.3, dec_dropout=0.3,
                 n_epochs=16, batch_size=64, lr=0.0004, seed=None):
        self.window_size = window_size
        self.d_emb = d_emb
        self.d_model = d_model
        self.nhead = nhead
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.d_ff = d_ff
        self.d_gru = d_gru
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if type(self.seed) is int:
            torch.manual_seed(self.seed)

    def _build_tensors(self, dataset):
        """Convert dataset patch arrays to tensors for DataLoader.

        Layout: [patches x K, dec_patches x K, patch_mask, patch_padding_mask]
        """
        K = dataset.num_attributes
        tensors = []
        for k in range(K):
            tensors.append(torch.LongTensor(dataset.patches[k]))
        for k in range(K):
            tensors.append(torch.LongTensor(dataset.decoder_patches[k]))
        tensors.append(torch.BoolTensor(dataset.patch_mask))
        tensors.append(torch.BoolTensor(dataset.patch_padding_mask))
        return tensors

    def _unpack_batch(self, batch, K):
        """Unpack a DataLoader batch into structured lists."""
        patches = [batch[k].to(self.device) for k in range(K)]
        dec_patches = [batch[K + k].to(self.device) for k in range(K)]
        patch_mask = batch[2 * K].to(self.device)
        padding_mask = batch[2 * K + 1].to(self.device)
        return patches, dec_patches, patch_mask, padding_mask

    def fit(self, dataset):
        """Train the PaCHITA model on the given dataset."""
        dataset._gen_patches(self.window_size)

        K = dataset.num_attributes
        attribute_dims = dataset.attribute_dims

        tensors = self._build_tensors(dataset)
        dataloader = DataLoader(
            TensorDataset(*tensors), batch_size=self.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )

        self.model = PaCHITANet(
            attribute_dims=attribute_dims,
            window_size=self.window_size,
            max_patches=dataset.max_len - self.window_size + 1,
            d_emb=self.d_emb, d_model=self.d_model, nhead=self.nhead,
            num_enc_layers=self.num_enc_layers, num_dec_layers=self.num_dec_layers,
            d_ff=self.d_ff, d_gru=self.d_gru,
            enc_dropout=self.enc_dropout, dec_dropout=self.dec_dropout,
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs, eta_min=self.lr / 10,
        )

        print("*" * 10 + "training" + "*" * 10)
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss, total_n = 0.0, 0

            for batch in tqdm(dataloader):
                patches, dec_patches, patch_mask, padding_mask = self._unpack_batch(batch, K)

                optimizer.zero_grad()
                logits_list = self.model(patches, dec_patches, padding_mask)

                loss = torch.tensor(0.0, device=self.device)
                mask_flat = patch_mask.reshape(-1)
                for k in range(K):
                    pred = logits_list[k].reshape(-1, attribute_dims[k] + 1)
                    true = patches[k].reshape(-1)
                    loss += F.cross_entropy(pred[~mask_flat], true[~mask_flat])

                n = patches[0].shape[0]
                total_loss += loss.item() * n
                total_n += n

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.)
                optimizer.step()

            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {total_loss / total_n:3f}]")
            scheduler.step()

        return self

    def detect(self, dataset):
        """Run anomaly detection. Returns (trace_scores, event_scores, attr_scores)."""
        if not hasattr(dataset, 'patches') or dataset.patches is None:
            dataset._gen_patches(self.window_size)

        K = dataset.num_attributes
        attribute_dims = dataset.attribute_dims
        T = dataset.max_len
        w = self.window_size
        P = T - w + 1

        dataloader = DataLoader(
            TensorDataset(*self._build_tensors(dataset)),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )

        self.model.eval()
        all_scores = []

        # Token position indices for overlap aggregation: (P, w)
        token_positions = (
            torch.arange(P, device=self.device).unsqueeze(1)
            + torch.arange(w, device=self.device).unsqueeze(0)
        )

        with torch.no_grad():
            print("*" * 10 + "detecting" + "*" * 10)
            for batch in tqdm(dataloader):
                patches, dec_patches, patch_mask, padding_mask = self._unpack_batch(batch, K)
                logits_list = self.model(patches, dec_patches, padding_mask)

                B = patches[0].shape[0]
                token_scores = torch.zeros(B, T, K, device=self.device)
                pos_flat = token_positions.reshape(-1).unsqueeze(0).expand(B, -1)

                for k in range(K):
                    probs = torch.softmax(logits_list[k], dim=-1)
                    true_probs = probs.gather(-1, patches[k].unsqueeze(-1)).squeeze(-1)
                    probs[probs <= true_probs.unsqueeze(-1)] = 0
                    scores = probs.sum(-1) * (~patch_mask).float()

                    token_scores[:, :, k].scatter_reduce_(
                        1, pos_flat, scores.reshape(B, -1), reduce='amax', include_self=True,
                    )

                all_scores.append(token_scores)

        attr_scores = torch.cat(all_scores, dim=0).cpu().numpy()
        trace_scores = attr_scores.max(axis=(1, 2))
        event_scores = attr_scores.max(axis=2)

        return trace_scores, event_scores, attr_scores


class PaCHITANet(nn.Module):
    """Patch-based multi-channel encoder-decoder network."""

    def __init__(self, attribute_dims, window_size, max_patches,
                 d_emb, d_model, nhead, num_enc_layers, num_dec_layers, d_ff, d_gru,
                 enc_dropout=0.1, dec_dropout=0.1):
        super().__init__()
        K = len(attribute_dims)
        self.num_channels = K

        emb_args = dict(d_emb=d_emb, d_model=d_model,
                        window_size=window_size, max_patches=max_patches)

        self.enc_embeddings = nn.ModuleList([
            PatchEmbedding(vocab_size=attribute_dims[k], **emb_args)
            for k in range(K)
        ])
        self.dec_embeddings = nn.ModuleList([
            PatchEmbedding(vocab_size=attribute_dims[k], **emb_args)
            for k in range(K)
        ])
        self.encoders = nn.ModuleList([
            ChannelEncoder(d_model=d_model, nhead=nhead, num_layers=num_enc_layers,
                           d_ff=d_ff, dropout=enc_dropout)
            for _ in range(K)
        ])
        self.channel_fusions = nn.ModuleList([
            SwiGLUFFN(d_in=K * d_model, d_out=d_model, d_ffn=d_ff)
            for _ in range(K)
        ])

        dec_args = dict(d_model=d_model, d_gru=d_gru, window_size=window_size,
                        num_dec_layers=num_dec_layers, dropout=dec_dropout)

        self.decoder_act = ChannelDecoderAct(vocab_size=attribute_dims[0], **dec_args)
        self.decoders_attr = nn.ModuleList([
            ChannelDecoderAttr(vocab_size=attribute_dims[k], **dec_args)
            for k in range(1, K)
        ])

    def forward(self, patches_list, decoder_patches_list, padding_mask=None):
        K = self.num_channels

        # Encode all channels
        enc_emb = [self.enc_embeddings[k](patches_list[k]) for k in range(K)]
        dec_emb = [self.dec_embeddings[k](decoder_patches_list[k]) for k in range(K)]
        enc_out = [self.encoders[k](enc_emb[k], padding_mask=padding_mask) for k in range(K)]

        # Cross-channel fusion: concat along hidden dim + per-decoder SwiGLU
        enc_cat = torch.cat(enc_out, dim=-1)  # (B, P, K*d_model)

        # Decode activity first, then attributes
        act_logits, hc_act = self.decoder_act(
            dec_emb[0], enc_out[0], self.channel_fusions[0](enc_cat),
            padding_mask=padding_mask, cross_padding_mask=padding_mask,
        )

        attr_logits = [
            self.decoders_attr[k - 1](
                dec_emb[k], enc_out[k], self.channel_fusions[k](enc_cat), hc_act,
                padding_mask=padding_mask, cross_padding_mask=padding_mask,
            )
            for k in range(1, K)
        ]

        return [act_logits] + attr_logits
