"""
Deep Wavelet Transformer (DWT)

Input:  (B, C, T)  — batch -> subcarriers -> time-steps
Output: (B, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


# Wavelet decomposition layer

class WaveletDecomposition(nn.Module):
    """
    Differentiable 1-D Discrete Wavelet Transform using fixed filter banks.
    -> concatenated approximation + detail coefficients.
    """

    def __init__(self, wavelet: str = "db4", level: int = 3):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        w = pywt.Wavelet(wavelet)

        # Low-pass (approx) and high-pass (detail) decomposition filters
        lo = torch.tensor(w.dec_lo, dtype=torch.float32).flip(0)
        hi = torch.tensor(w.dec_hi, dtype=torch.float32).flip(0)
        self.filter_len = len(lo)

        # Shape
        self.register_buffer("lo_filter", lo.unsqueeze(0).unsqueeze(0))
        self.register_buffer("hi_filter", hi.unsqueeze(0).unsqueeze(0))

    def _dwt1d(self, x: torch.Tensor):
        """Single-level 1-D DWT"""
        B, C, T = x.shape
        pad = self.filter_len - 1

        # Merge batch & channel for grouped conv
        x_flat = x.reshape(B * C, 1, T)
        x_pad  = F.pad(x_flat, (pad // 2, pad - pad // 2), mode="reflect")

        lo_f = self.lo_filter.to(x.device)
        hi_f = self.hi_filter.to(x.device)

        approx = F.conv1d(x_pad, lo_f, stride=2).reshape(B, C, -1)
        detail = F.conv1d(x_pad, hi_f, stride=2).reshape(B, C, -1)
        return approx, detail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: (B, C, T)
        -> (B, C * (level+1), T') - concatenated wavelet coefficients
        """
        coeffs = []
        cur = x
        for _ in range(self.level):
            approx, detail = self._dwt1d(cur)
            coeffs.append(detail)
            cur = approx
        coeffs.append(cur)   # final approximation

        min_len = min(c.shape[-1] for c in coeffs)
        out = [F.adaptive_avg_pool1d(c, min_len) for c in coeffs]
        return torch.cat(out, dim=1)  # (B, C*(level+1), min_len)


# Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# Channel Attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = self.fc(x.mean(-1))   # (B, C)
        return x * w.unsqueeze(-1)


# DWT-Transformer Block

class DWTTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h  = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x  = x + h
        x  = x + self.ff(self.norm2(x))
        return x


# ─── Full Model ───────────────────────────────────────────────────────────────

class DeepWaveletTransformer(nn.Module):
    """
    Deep Wavelet Transformer

    Staege:
      1. Wavelet decomposition → multi-scale features
      2. Channel attention
      3. Linear projection → embedding space
      4. Positional encoding
      5. N × Transformer blocks
      6. Global average pooling + classifier head
    """

    def __init__(
        self,
        in_channels:  int = 30,      # number of WiFi subcarriers
        num_classes:  int = 2,
        embed_dim:    int = 64,
        num_heads:    int = 4,
        num_layers:   int = 3,
        dropout:      float = 0.1,
        wavelet:      str = "db4",
        wavelet_level: int = 3,
    ):
        super().__init__()

        self.wavelet_decomp = WaveletDecomposition(wavelet, wavelet_level)

        wavelet_channels = in_channels * (wavelet_level + 1)
        self.channel_attn = ChannelAttention(wavelet_channels)

        self.input_proj = nn.Sequential(
            nn.Conv1d(wavelet_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(embed_dim, dropout=dropout)

        self.transformer = nn.Sequential(
            *[DWTTransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)  — CSI amplitude/phase matrix
        Returns:
            logits: (B, num_classes)
        """
        # 1. Wavelet decomposition
        x = self.wavelet_decomp(x)            # (B, C*(level+1), T')

        # 2. Channel attention
        x = self.channel_attn(x)             # (B, C*(level+1), T')

        # 3. Project to embedding space
        x = self.input_proj(x)               # (B, embed_dim, T')

        # 4. Prepare for transformer: (B, T', embed_dim)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)

        # 5. Transformer blocks
        for block in self.transformer:
            x = block(x)

        # 6. Global avg pool + classify
        x = x.mean(dim=1)                    # (B, embed_dim)
        return self.classifier(x)            # (B, num_classes)


# ─── Baseline CNN (for comparison) ───────────────────────────────────────────

class BaselineCNN(nn.Module):
    """Simple 1-D CNN baseline"""

    def __init__(self, in_channels: int = 30, num_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64,  kernel_size=7, padding=3), nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,          128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128,         256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# Factory

def get_model(name: str = "DWT", **kwargs) -> nn.Module:
    models = {
        "DWT":         DeepWaveletTransformer,
        "Transformer": DeepWaveletTransformer,
        "CNN":         BaselineCNN,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models.keys())}")
    return models[name](**kwargs)


if __name__ == "__main__":
    import config
    model = get_model(
        config.MODEL_NAME,
        in_channels=config.NUM_SUBCARRIERS,
        num_classes=config.NUM_CLASSES,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        wavelet=config.WAVELET,
        wavelet_level=config.WAVELET_LEVEL,
    )
    dummy = torch.randn(4, config.NUM_SUBCARRIERS, config.WINDOW_SIZE)
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,}")
