"""Shape and wiring tests for the model components."""
from __future__ import annotations

import torch

from reboot.models import (
    HybridEncoder,
    ProjectionHead,
    ReBootModel,
    ReconstructionDecoder,
    RegressionHead,
    TCNEncoder,
    TransformerEncoder,
    build_encoder,
)


def test_tcn_encoder_output_shape():
    enc = TCNEncoder(in_channels=4, hidden=128)
    x = torch.randn(2, 128, 4)
    h = enc(x)
    assert h.shape == (2, 128)


def test_transformer_encoder_output_shape():
    enc = TransformerEncoder(in_channels=4, hidden=128, max_len=128)
    x = torch.randn(2, 128, 4)
    h = enc(x)
    assert h.shape == (2, 128)


def test_hybrid_encoder_output_shape():
    enc = HybridEncoder(in_channels=4, hidden=128)
    x = torch.randn(2, 128, 4)
    h = enc(x)
    assert h.shape == (2, 128)


def test_build_encoder_dispatch():
    for name, klass in [("tcn", TCNEncoder), ("transformer", TransformerEncoder),
                        ("hybrid", HybridEncoder)]:
        enc = build_encoder(name, in_channels=4)
        assert isinstance(enc, klass)


def test_projection_head_shape():
    p = ProjectionHead(128, 64)
    h = torch.randn(5, 128)
    z = p(h)
    assert z.shape == (5, 64)


def test_reconstruction_decoder_shape():
    dec = ReconstructionDecoder(z_dim=128, out_channels=4, W=128, hidden=128)
    z = torch.randn(3, 128)
    out = dec(z)
    assert out.shape == (3, 128, 4)


def test_regression_head_output_in_unit_interval():
    head = RegressionHead(in_dim=128)
    z = torch.randn(8, 128)
    y = head(z)
    assert y.shape == (8,)
    assert (y >= 0).all() and (y <= 1).all()


def test_reboot_model_end_to_end():
    model = ReBootModel(in_channels=4, W=128, encoder_type="hybrid",
                        hidden=128, d_s=64, d_d=64)
    x = torch.randn(2, 128, 4)
    z_s, z_d = model.encode(x)
    assert z_s.shape == (2, 64)
    assert z_d.shape == (2, 64)
    x_hat = model.reconstruct(z_s, z_d)
    assert x_hat.shape == x.shape
    y_hat = model.predict_soc(z_s, z_d)
    assert y_hat.shape == (2,)
    assert (y_hat >= 0).all() and (y_hat <= 1).all()


def test_reboot_model_single_latent_ablation():
    model = ReBootModel(in_channels=4, W=64, hidden=64, d_s=32, d_d=32,
                        single_latent=True)
    x = torch.randn(2, 64, 4)
    z_s, z_d = model.encode(x)
    assert z_s.shape == (2, 64)        # combined
    assert z_d.shape == (2, 0)         # empty placeholder
    x_hat = model.reconstruct(z_s, z_d)
    assert x_hat.shape == x.shape
