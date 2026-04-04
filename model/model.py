'''
PatchBPAD: Patch-based Business Process Anomaly Detection

@author: Y.J. Lee
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.compo import PatchEmbedding, ChannelEncoder, ChannelDecoderAct, ChannelDecoderAttr


class PatchBPAD:
    """Wrapper class with fit/detect interface for PatchBPAD anomaly detection."""

    name = 'PatchBPAD'

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
        """Convert dataset patch arrays to tensors for DataLoader."""
        K = dataset.num_attributes
        tensors = []
        # Encoder patches (also used as targets) for each channel
        for k in range(K):
            tensors.append(torch.LongTensor(dataset.patches[k]))
        # Decoder patches for each channel
        for k in range(K):
            tensors.append(torch.LongTensor(dataset.decoder_patches[k]))
        # Patch mask (shared) — per-position within patches
        tensors.append(torch.BoolTensor(dataset.patch_mask))
        # Patch padding mask (shared) — per-patch, True where entirely padding
        tensors.append(torch.BoolTensor(dataset.patch_padding_mask))
        return tensors

    def _unpack_batch(self, batch, K):
        """Unpack a batch from the DataLoader into structured lists.

        Layout: [patches_0..K-1, dec_patches_0..K-1, patch_mask, patch_padding_mask]
        """
        patches_list = [batch[k].to(self.device) for k in range(K)]
        decoder_patches_list = [batch[K + k].to(self.device) for k in range(K)]
        patch_mask = batch[2 * K].to(self.device)
        patch_padding_mask = batch[2 * K + 1].to(self.device)
        return patches_list, decoder_patches_list, patch_mask, patch_padding_mask

    def fit(self, dataset):
        """Train the PatchBPAD model on the given dataset.

        Args:
            dataset: Dataset object with features, attribute_dims, etc.

        Returns:
            self
        """
        # 1. Generate patches
        dataset._gen_patches(self.window_size)

        K = dataset.num_attributes
        attribute_dims = dataset.attribute_dims

        # 2. Build tensors
        tensors = self._build_tensors(dataset)
        tensor_dataset = TensorDataset(*tensors)
        dataloader = DataLoader(tensor_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=4, pin_memory=True,
                                drop_last=True)

        # 4. Instantiate model
        max_patches = dataset.max_len - self.window_size + 1
        self.model = PatchBPADNet(
            attribute_dims=attribute_dims,
            window_size=self.window_size,
            max_patches=max_patches,
            d_emb=self.d_emb,
            d_model=self.d_model,
            nhead=self.nhead,
            num_enc_layers=self.num_enc_layers,
            num_dec_layers=self.num_dec_layers,
            d_ff=self.d_ff,
            d_gru=self.d_gru,
            enc_dropout=self.enc_dropout,
            dec_dropout=self.dec_dropout,
        )
        self.model.to(self.device)

        # 5. Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=self.lr / 10)

        # 6. Training loop
        print("*" * 10 + "training" + "*" * 10)
        self.model.train()
        for epoch in range(self.n_epochs):
            train_loss = 0.0
            train_num = 0

            for batch in tqdm(dataloader):
                patches_list, decoder_patches_list, patch_mask, patch_padding_mask = \
                    self._unpack_batch(batch, K)

                optimizer.zero_grad()

                # Forward pass
                logits_list = self.model(patches_list, decoder_patches_list, patch_padding_mask)

                # Compute loss: cross-entropy on non-masked positions, summed across channels
                loss = torch.tensor(0.0, device=self.device)
                mask_flat = patch_mask.reshape(-1)  # (batch * num_patches * w)

                for k in range(K):
                    pred = logits_list[k].reshape(-1, attribute_dims[k] + 1)  # (B*P*W, V+1)
                    true = patches_list[k].reshape(-1)  # (B*P*W)
                    # CE only on non-masked positions
                    loss += F.cross_entropy(pred[~mask_flat], true[~mask_flat])

                batch_size_actual = patches_list[0].shape[0]
                train_loss += loss.item() * batch_size_actual
                train_num += batch_size_actual

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.)
                optimizer.step()

            train_loss_epoch = train_loss / train_num
            print(f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]")
            scheduler.step()

        return self

    def detect(self, dataset):
        """Run anomaly detection on the given dataset.

        Args:
            dataset: Dataset object (may be same as training dataset or a new one)

        Returns:
            tuple: (trace_level_scores, event_level_scores, attr_level_scores)
                - trace_level_scores:  (N,) numpy array
                - event_level_scores:  (N, max_case_len) numpy array
                - attr_level_scores:   (N, max_case_len, K) numpy array
        """
        # 1. Generate patches if not already done
        if not hasattr(dataset, 'patches') or dataset.patches is None:
            dataset._gen_patches(self.window_size)

        K = dataset.num_attributes
        attribute_dims = dataset.attribute_dims
        max_case_len = dataset.max_len
        w = self.window_size

        # 2. Build tensors (no shuffle)
        tensors = self._build_tensors(dataset)
        tensor_dataset = TensorDataset(*tensors)
        dataloader = DataLoader(tensor_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)

        # 3. Evaluate
        self.model.eval()
        all_scores = []

        # Precompute token position indices for overlap aggregation: (num_patches, w)
        num_patches = max_case_len - w + 1
        patch_idx = torch.arange(num_patches, device=self.device).unsqueeze(1)
        win_idx = torch.arange(w, device=self.device).unsqueeze(0)
        token_positions = (patch_idx + win_idx)  # (num_patches, w)

        with torch.no_grad():
            print("*" * 10 + "detecting" + "*" * 10)
            for batch in tqdm(dataloader):
                patches_list, decoder_patches_list, patch_mask, patch_padding_mask = \
                    self._unpack_batch(batch, K)

                # Forward pass
                logits_list = self.model(patches_list, decoder_patches_list, patch_padding_mask)

                batch_size_actual = patches_list[0].shape[0]

                # token_scores: (batch, max_case_len, K)
                token_scores = torch.zeros(batch_size_actual, max_case_len, K,
                                           device=self.device)

                # Expand position indices for batch: (batch, num_patches * w)
                pos_flat = token_positions.reshape(-1).unsqueeze(0).expand(batch_size_actual, -1)

                for k in range(K):
                    # Per-position anomaly scoring at patch level
                    probs = torch.softmax(logits_list[k], dim=-1)  # (B, P, W, V+1)
                    targets = patches_list[k]  # (B, P, W)
                    true_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, P, W)
                    probs[probs <= true_probs.unsqueeze(-1)] = 0
                    patch_scores = probs.sum(-1)  # (B, P, W)
                    patch_scores = patch_scores * (~patch_mask).float()

                    # Vectorized overlap aggregation via scatter_reduce (max)
                    scores_flat = patch_scores.reshape(batch_size_actual, -1)  # (B, P*W)
                    token_scores[:, :, k].scatter_reduce_(
                        1, pos_flat, scores_flat, reduce='amax', include_self=True
                    )

                all_scores.append(token_scores)

        # 5. Concatenate all batches
        attr_level_abnormal_scores = torch.cat(all_scores, dim=0).cpu().numpy()
        # (N, max_case_len, K)

        trace_level_abnormal_scores = attr_level_abnormal_scores.max(axis=(1, 2))
        event_level_abnormal_scores = attr_level_abnormal_scores.max(axis=2)

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores



class PatchBPADNet(nn.Module):
    """Patch-based multi-channel encoder-decoder network for anomaly detection."""

    def __init__(self, attribute_dims, window_size, max_patches,
                 d_emb, d_model, nhead, num_enc_layers, num_dec_layers, d_ff, d_gru,
                 enc_dropout=0.1, dec_dropout=0.1):
        """
        Args:
            attribute_dims: list/array of K vocab sizes (one per channel)
            window_size:    patch width w
            max_patches:    maximum number of patches in a sequence
            d_emb:          token embedding dimension
            d_model:        model hidden dimension
            nhead:          number of attention heads
            num_enc_layers: number of transformer encoder layers
            num_dec_layers: number of GRU layers in decoders
            d_ff:           feed-forward dimension in encoder
            d_gru:          GRU hidden dimension in decoders
            enc_dropout:    dropout rate for encoder
            dec_dropout:    dropout rate for decoders
        """
        super().__init__()
        self.num_channels = len(attribute_dims)
        self.attribute_dims = list(attribute_dims)

        # One PatchEmbedding per channel for encoder
        self.enc_patch_embeddings = nn.ModuleList([
            PatchEmbedding(vocab_size=attribute_dims[k], d_emb=d_emb,
                           d_model=d_model, window_size=window_size,
                           max_patches=max_patches)
            for k in range(self.num_channels)
        ])

        # One ChannelEncoder per channel
        self.encoders = nn.ModuleList([
            ChannelEncoder(d_model=d_model, nhead=nhead, num_layers=num_enc_layers, d_ff=d_ff, dropout=enc_dropout)
            for k in range(self.num_channels)
        ])


        # Separate PatchEmbedding per channel for decoder
        self.dec_patch_embeddings = nn.ModuleList([
            PatchEmbedding(vocab_size=attribute_dims[k], d_emb=d_emb,
                           d_model=d_model, window_size=window_size,
                           max_patches=max_patches)
            for k in range(self.num_channels)
        ])

        # Activity decoder (channel 0)
        self.decoder_act = ChannelDecoderAct(
            vocab_size=attribute_dims[0], d_model=d_model,
            d_gru=d_gru, window_size=window_size,
            num_dec_layers=num_dec_layers, dropout=dec_dropout
        )

        # Attribute decoders (channels 1..K-1)
        self.decoders_attr = nn.ModuleList([
            ChannelDecoderAttr(
                vocab_size=attribute_dims[k], d_model=d_model,
                d_gru=d_gru, window_size=window_size,
                num_dec_layers=num_dec_layers, dropout=dec_dropout
            )
            for k in range(1, self.num_channels)
        ])

    def forward(self, patches_list, decoder_patches_list, padding_mask=None):
        """
        Args:
            patches_list:         list of K LongTensors (batch, num_patches, w) -- encoder input
            decoder_patches_list: list of K LongTensors (batch, num_patches, w) -- decoder input
            padding_mask:         BoolTensor (batch, num_patches) -- True where patch is entirely padding

        Returns:
            list of K tensors (batch, num_patches, w, vocab_size+1)
        """
        K = self.num_channels

        # 1. Encoder embeddings
        enc_emb = [self.enc_patch_embeddings[k](patches_list[k]) for k in range(K)]

        # 2. Decoder embeddings (separate embedding layer per channel)
        dec_emb = [self.dec_patch_embeddings[k](decoder_patches_list[k]) for k in range(K)]

        # 3. Encode (mask out padding patches from self-attention)
        enc_out = [self.encoders[k](enc_emb[k], padding_mask=padding_mask) for k in range(K)]

        # 4. Activity decode (channel 0)
        act_logits, hc_act = self.decoder_act(dec_emb[0], enc_out[0], padding_mask=padding_mask)

        # 5. Attribute decode (channels 1..K-1)
        attr_logits = []
        for k in range(1, K):
            logits_k = self.decoders_attr[k - 1](dec_emb[k], enc_out[k], hc_act, padding_mask=padding_mask)
            attr_logits.append(logits_k)

        # 6. Return all channel logits
        return [act_logits] + attr_logits


