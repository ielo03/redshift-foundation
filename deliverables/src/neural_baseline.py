from __future__ import annotations

import torch
from torch import nn


class SpectraConvAutoencoderWithRedshiftHead(nn.Module):
    """Small spectra-only convolutional baseline.

    It is intentionally much simpler than the final AION-style transformer,
    but it is a real neural baseline that jointly learns:

    - spectrum reconstruction
    - redshift prediction
    """

    def __init__(self, input_length: int, latent_dim: int = 256):
        super().__init__()
        self.input_length = input_length
        hidden_dim = max(256, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_length),
        )
        self.redshift_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        recon = self.decoder(z)
        redshift = self.redshift_head(z).squeeze(-1)
        return recon, redshift


class SpectraTransformerWithRedshiftHead(nn.Module):
    """Spectra-only transformer baseline.

    This is closer to the assignment framing than the tiny conv baseline.
    It embeds the spectrum into patches, applies a transformer encoder, and
    predicts both masked-spectrum reconstruction and redshift.
    """

    def __init__(
        self,
        input_length: int,
        patch_size: int = 61,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        input_channels: int = 1,
    ):
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.padded_length = ((input_length + patch_size - 1) // patch_size) * patch_size
        self.num_patches = self.padded_length // patch_size
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_size * input_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size),
        )
        self.redshift_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (B, L) or (B, 1, L), got {tuple(x.shape)}")
        length = x.shape[-1]
        if length == self.padded_length:
            return x
        if length > self.padded_length:
            return x[..., : self.padded_length]
        pad = self.padded_length - length
        return torch.nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._pad_to_patch_multiple(x)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(x.shape[0], self.num_patches, self.input_channels * self.patch_size)
        tokens = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        patch_recon = self.recon_head(encoded[:, 1:, :])
        recon = patch_recon.reshape(x.shape[0], -1)
        recon = recon[:, : self.input_length]
        redshift = self.redshift_head(encoded[:, 0, :]).squeeze(-1)
        return recon, redshift


class SpectraTransformerWithRedshiftToken(nn.Module):
    """Patch-token spectra transformer with an explicit redshift token.

    The sequence is:

    - one learned redshift query token
    - one token per spectrum patch

    This keeps redshift prediction inside the transformer sequence instead of
    treating it as a post-hoc side head on a generic CLS embedding.
    """

    def __init__(
        self,
        input_length: int,
        patch_size: int = 61,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        input_channels: int = 1,
    ):
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.padded_length = ((input_length + patch_size - 1) // patch_size) * patch_size
        self.num_patches = self.padded_length // patch_size
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_size * input_channels, d_model)
        self.redshift_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size),
        )
        self.redshift_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (B, L) or (B, C, L), got {tuple(x.shape)}")
        length = x.shape[-1]
        if length == self.padded_length:
            return x
        if length > self.padded_length:
            return x[..., : self.padded_length]
        pad = self.padded_length - length
        return torch.nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._pad_to_patch_multiple(x)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(x.shape[0], self.num_patches, self.input_channels * self.patch_size)
        tokens = self.patch_embed(patches)
        redshift_tokens = self.redshift_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([redshift_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        patch_recon = self.recon_head(encoded[:, 1:, :])
        recon = patch_recon.reshape(x.shape[0], -1)
        recon = recon[:, : self.input_length]
        redshift = self.redshift_head(encoded[:, 0, :]).squeeze(-1)
        return recon, redshift


class SpectraTransformerWithCoarseRedshiftHead(nn.Module):
    """Raw-spectrum transformer with a coarse classification + fine regression redshift head."""

    def __init__(
        self,
        input_length: int,
        num_bins: int,
        patch_size: int = 61,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.padded_length = ((input_length + patch_size - 1) // patch_size) * patch_size
        self.num_patches = self.padded_length // patch_size
        self.d_model = d_model
        self.num_bins = num_bins

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size),
        )
        self.coarse_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_bins),
        )
        self.fine_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (B, L) or (B, 1, L), got {tuple(x.shape)}")
        length = x.shape[-1]
        if length == self.padded_length:
            return x
        if length > self.padded_length:
            return x[..., : self.padded_length]
        pad = self.padded_length - length
        return torch.nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._pad_to_patch_multiple(x)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        patches = patches.squeeze(1)
        tokens = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        patch_recon = self.recon_head(encoded[:, 1:, :])
        recon = patch_recon.reshape(x.shape[0], -1)
        recon = recon[:, : self.input_length]
        coarse_logits = self.coarse_head(encoded[:, 0, :])
        fine_residual = torch.tanh(self.fine_head(encoded[:, 0, :]).squeeze(-1))
        return recon, coarse_logits, fine_residual


class SpectraTransformerWithRedshiftTokenBins(nn.Module):
    """Redshift-token foundation variant with bin classification + residual."""

    def __init__(
        self,
        input_length: int,
        num_bins: int,
        patch_size: int = 61,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        input_channels: int = 1,
    ):
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.padded_length = ((input_length + patch_size - 1) // patch_size) * patch_size
        self.num_patches = self.padded_length // patch_size
        self.num_bins = num_bins

        self.patch_embed = nn.Linear(patch_size * input_channels, d_model)
        self.redshift_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size),
        )
        self.bin_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_bins),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (B, L) or (B, C, L), got {tuple(x.shape)}")
        length = x.shape[-1]
        if length == self.padded_length:
            return x
        if length > self.padded_length:
            return x[..., : self.padded_length]
        pad = self.padded_length - length
        return torch.nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._pad_to_patch_multiple(x)
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(x.shape[0], self.num_patches, self.input_channels * self.patch_size)
        tokens = self.patch_embed(patches)
        redshift_tokens = self.redshift_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([redshift_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        redshift_repr = encoded[:, 0, :]
        patch_recon = self.recon_head(encoded[:, 1:, :])
        recon = patch_recon.reshape(x.shape[0], -1)
        recon = recon[:, : self.input_length]
        bin_logits = self.bin_head(redshift_repr)
        residual = torch.tanh(self.residual_head(redshift_repr).squeeze(-1))
        return recon, bin_logits, residual


class AIONCodecMaskedTransformer(nn.Module):
    """Legacy discrete-token transformer for DESI spectra + redshift.

    This model expects tokenized inputs from the local AION codec path:

    - 273 spectrum tokens for `tok_spectrum_desi`
    - 1 legacy redshift token path from the original AION codec

    It performs masked-token prediction over the discrete spectrum tokens and
    the redshift token, matching the assignment's codec-based framing more
    closely than the patch-based fallback baseline.
    """

    def __init__(
        self,
        seq_len: int,
        input_vocab_size: int = 1025,
        output_vocab_size: int = 1024,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.token_embed = nn.Embedding(input_vocab_size, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_vocab_size),
        )

    def forward(self, input_ids: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, L), got {tuple(input_ids.shape)}")
        if token_types.shape != input_ids.shape:
            raise ValueError(
                f"token_types shape {tuple(token_types.shape)} must match input_ids shape {tuple(input_ids.shape)}"
            )

        x = self.token_embed(input_ids) + self.type_embed(token_types)
        x = x + self.pos_embed[:, : x.shape[1], :]
        encoded = self.encoder(x)
        logits = self.lm_head(encoded)
        return logits


class AIONCodecRedshiftAwareTransformer(nn.Module):
    """Legacy codec transformer with a dedicated continuous redshift head.

    This variant keeps the masked-token reconstruction path, but it also
    predicts redshift directly from a learned summary token so the model is not
    forced to recover z only through a discrete scalar codec.
    """

    def __init__(
        self,
        seq_len: int,
        input_vocab_size: int = 1025,
        output_vocab_size: int = 1024,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len + 1
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.token_embed = nn.Embedding(input_vocab_size, d_model)
        self.type_embed = nn.Embedding(3, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_vocab_size),
        )
        self.redshift_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, input_ids: torch.Tensor, token_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, L), got {tuple(input_ids.shape)}")
        if token_types.shape != input_ids.shape:
            raise ValueError(
                f"token_types shape {tuple(token_types.shape)} must match input_ids shape {tuple(input_ids.shape)}"
            )

        cls_tokens = self.cls_token.expand(input_ids.shape[0], -1, -1)
        cls_types = torch.full((input_ids.shape[0], 1), 2, dtype=torch.long, device=input_ids.device)
        x = self.token_embed(input_ids) + self.type_embed(token_types)
        x = torch.cat([cls_tokens + self.type_embed(cls_types), x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        encoded = self.encoder(x)
        logits = self.lm_head(encoded[:, 1:, :])
        redshift = self.redshift_head(encoded[:, 0, :]).squeeze(-1)
        return logits, redshift


class AIONSpectrumMaskedTransformerWithRedshiftHead(nn.Module):
    """Spectra-token transformer trained from scratch with a continuous redshift head.

    This is the cleaner project baseline:

    - spectrum tokens come from the AION DESI spectrum tokenizer
    - redshift is predicted as a continuous scalar
    - validation and test masking can be fixed and deterministic
    """

    def __init__(
        self,
        seq_len: int,
        input_vocab_size: int = 1025,
        output_vocab_size: int = 1024,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        self.token_embed = nn.Embedding(input_vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_vocab_size),
        )
        self.redshift_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, L), got {tuple(input_ids.shape)}")

        token_embeddings = self.token_embed(input_ids)
        cls_tokens = self.cls_token.expand(input_ids.shape[0], -1, -1)
        x = torch.cat([cls_tokens, token_embeddings], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        encoded = self.encoder(x)
        logits = self.recon_head(encoded[:, 1:, :])
        redshift = self.redshift_head(encoded[:, 0, :]).squeeze(-1)
        return logits, redshift
