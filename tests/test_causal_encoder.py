"""Tests for causal context encoder refactor (ViTBase, build_frame_causal_mask)."""

import pytest
import torch

from eb_jepa.models.components import build_frame_causal_mask
from eb_jepa.models.encoders import (
    ViTCLSEncoder,
    ViTEncoder,
)

B, C, T, IMG = 2, 3, 4, 64
PATCH = 16
G = IMG // PATCH  # 4


class TestViTEncoderNoCausal:
    """ViTEncoder with context_frames=0 (default, backward compat)."""

    def test_shape_5d(self):
        enc = ViTEncoder(scale="tiny", patch_size=PATCH, image_size=IMG)
        x = torch.randn(B, C, T, IMG, IMG)
        out = enc(x)
        assert out.shape == (B, 192, T, G, G)

    def test_shape_4d(self):
        enc = ViTEncoder(scale="tiny", patch_size=PATCH, image_size=IMG)
        x = torch.randn(B, C, IMG, IMG)
        out = enc(x)
        assert out.shape == (B, 192, G, G)

    def test_output_dim_projection(self):
        enc = ViTEncoder(scale="tiny", patch_size=PATCH, image_size=IMG, output_dim=64)
        x = torch.randn(B, C, T, IMG, IMG)
        out = enc(x)
        assert out.shape == (B, 64, T, G, G)


class TestViTEncoderCausal:
    """ViTEncoder with context_frames > 0."""

    def test_shape(self):
        enc = ViTEncoder(
            scale="tiny",
            patch_size=PATCH,
            image_size=IMG,
            context_frames=3,
            causal_depth=2,
        )
        x = torch.randn(B, C, T, IMG, IMG)
        out = enc(x)
        assert out.shape == (B, 192, T, G, G)

    def test_output_dim_projection(self):
        enc = ViTEncoder(
            scale="tiny",
            patch_size=PATCH,
            image_size=IMG,
            output_dim=64,
            context_frames=3,
            causal_depth=2,
        )
        x = torch.randn(B, C, T, IMG, IMG)
        out = enc(x)
        assert out.shape == (B, 64, T, G, G)

    def test_causality_frame0(self):
        """Frame 0 encoding must be identical regardless of future frames."""
        torch.manual_seed(42)
        enc = ViTEncoder(
            scale="tiny",
            patch_size=PATCH,
            image_size=IMG,
            context_frames=3,
            causal_depth=2,
        )
        enc.eval()

        x1 = torch.randn(1, C, T, IMG, IMG)
        x2 = x1.clone()
        x2[:, :, 1:] = torch.randn(1, C, T - 1, IMG, IMG)

        with torch.no_grad():
            out1 = enc(x1)
            out2 = enc(x2)

        torch.testing.assert_close(out1[:, :, 0], out2[:, :, 0])


class TestViTCLSEncoderCausal:
    """ViTCLSEncoder with context_frames > 0."""

    def test_shape(self):
        enc = ViTCLSEncoder(
            scale="tiny",
            patch_size=PATCH,
            image_size=IMG,
            context_frames=3,
            causal_depth=2,
        )
        x = torch.randn(B, C, T, IMG, IMG)
        out = enc(x)
        assert out.shape == (B, 192, T, 1, 1)

    def test_shape_no_causal(self):
        enc = ViTCLSEncoder(scale="tiny", patch_size=PATCH, image_size=IMG)
        x = torch.randn(B, C, T, IMG, IMG)
        out = enc(x)
        assert out.shape == (B, 192, T, 1, 1)


class TestBuildFrameCausalMask:
    """build_frame_causal_mask backward compat and context_window."""

    def test_full_causal(self):
        mask = build_frame_causal_mask(3, 2, 2)
        assert mask.shape == (12, 12)
        assert mask[0, 0].item() is True
        assert mask[0, 4].item() is False  # frame 0 cannot see frame 1
        assert mask[4, 0].item() is True  # frame 1 can see frame 0

    def test_context_window(self):
        mask = build_frame_causal_mask(4, 1, 1, context_window=1)
        # frame 0 sees only itself
        assert mask[0, 0].item() is True
        # frame 1 sees frame 0 and itself
        assert mask[1, 0].item() is True
        assert mask[1, 1].item() is True
        # frame 2 sees frames 1-2 but NOT frame 0
        assert mask[2, 0].item() is False
        assert mask[2, 1].item() is True
        assert mask[2, 2].item() is True
        # frame 3 sees frames 2-3 but NOT frames 0-1
        assert mask[3, 0].item() is False
        assert mask[3, 1].item() is False
        assert mask[3, 2].item() is True
        assert mask[3, 3].item() is True

    def test_backward_compat_no_context_window(self):
        """Calling without context_window matches old behavior."""
        mask_new = build_frame_causal_mask(3, 2, 2)
        T, H, W = 3, 2, 2
        HW = H * W
        N = T * HW
        mask_ref = torch.zeros(N, N, dtype=torch.bool)
        block = torch.ones(HW, HW, dtype=torch.bool)
        for t1 in range(T):
            for t2 in range(t1 + 1):
                mask_ref[t1 * HW : (t1 + 1) * HW, t2 * HW : (t2 + 1) * HW] = block
        assert torch.equal(mask_new, mask_ref)
