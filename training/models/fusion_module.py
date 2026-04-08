# training/models/fusion_module.py
# ── NOVEL COMPONENT #1: Cross-Modal Fusion Layer ──
#
# Architecture: Scaled Dot-Product Cross-Attention
#   - Query  = text embedding (primary modality)
#   - Key    = [text, image, audio] embeddings (all modalities)
#   - Value  = [text, image, audio] embeddings
# Output: unified 384-dim multimodal representation
#
# This is trainable end-to-end and novel in the educational AI context.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModalityProjector(nn.Module):
    """Projects variable-dim modality embedding → unified dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.proj(x)


class CrossModalAttention(nn.Module):
    """
    3-head cross-attention fusion.
    Q = text, K/V = concatenation of all available modalities.
    """
    def __init__(self, dim: int = 384, num_heads: int = 3, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, keys: list[torch.Tensor]) -> torch.Tensor:
        """
        query : (B, dim)   — text embedding
        keys  : list of (B, dim) — [text, image, audio] as available
        """
        B = query.size(0)
        kv = torch.stack(keys, dim=1)      # (B, M, dim)  M = num modalities

        Q = self.q_proj(query).unsqueeze(1)  # (B, 1, dim)
        K = self.k_proj(kv)                  # (B, M, dim)
        V = self.v_proj(kv)                  # (B, M, dim)

        # Split heads
        def split(t, seq):
            return t.view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split(Q, 1)
        K = split(K, kv.size(1))
        V = split(V, kv.size(1))

        # Scaled dot-product attention
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, 1, M)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        ctx = torch.matmul(attn, V).squeeze(2)   # (B, H, head_dim)
        ctx = ctx.contiguous().view(B, self.dim)  # (B, dim)

        out = self.out_proj(ctx)
        return self.norm(out + query)              # residual connection


class MultiModalFusion(nn.Module):
    """
    Full fusion module combining text (384), image (512→384), audio (384).
    Only modalities that are present are fused; missing ones are skipped.
    """
    TEXT_DIM  = 384
    IMAGE_DIM = 512   # CLIP ViT-B/32 output
    AUDIO_DIM = 384   # Whisper transcription → text encoder

    def __init__(self, unified_dim: int = 384):
        super().__init__()
        self.image_proj = ModalityProjector(self.IMAGE_DIM, unified_dim)
        self.attn       = CrossModalAttention(dim=unified_dim, num_heads=3)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(unified_dim, unified_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(unified_dim * 2, unified_dim),
            nn.LayerNorm(unified_dim),
        )

    def forward(
        self,
        text_emb  : torch.Tensor,                    # (B, 384) — required
        image_emb : torch.Tensor | None = None,      # (B, 512) — optional
        audio_emb : torch.Tensor | None = None,      # (B, 384) — optional
    ) -> torch.Tensor:
        keys = [text_emb]

        if image_emb is not None:
            keys.append(self.image_proj(image_emb))

        if audio_emb is not None:
            keys.append(audio_emb)

        fused = self.attn(text_emb, keys)
        return self.mlp(fused)


# ── Utility: save / load ─────────────────────────────────────────────
def save_fusion(model: MultiModalFusion, path: str = "weights/fusion_module.pt"):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[fusion] Saved → {path}")

def load_fusion(path: str = "weights/fusion_module.pt") -> MultiModalFusion:
    model = MultiModalFusion()
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"[fusion] Loaded fine-tuned weights from {path}")
    except FileNotFoundError:
        print("[fusion] No fine-tuned weights found — using randomly initialized fusion.")
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = MultiModalFusion()
    text  = torch.randn(4, 384)
    image = torch.randn(4, 512)
    audio = torch.randn(4, 384)

    out_text_only  = model(text)
    out_multimodal = model(text, image, audio)

    print(f"Text-only output:    {out_text_only.shape}")    # (4, 384)
    print(f"Multimodal output:   {out_multimodal.shape}")   # (4, 384)
    print("✅ Fusion module sanity check passed.")
