from reboot.models.encoders.hybrid import HybridEncoder
from reboot.models.encoders.tcn import TCNEncoder
from reboot.models.encoders.transformer import TransformerEncoder


def build_encoder(encoder_type: str, in_channels: int, hidden: int = 128, **kwargs):
    """Factory for encoder variants. `encoder_type` ∈ {tcn, transformer, hybrid}."""
    et = encoder_type.lower()
    if et == "tcn":
        return TCNEncoder(in_channels=in_channels, hidden=hidden, **kwargs)
    if et == "transformer":
        return TransformerEncoder(in_channels=in_channels, hidden=hidden, **kwargs)
    if et == "hybrid":
        return HybridEncoder(in_channels=in_channels, hidden=hidden, **kwargs)
    raise ValueError(f"Unknown encoder_type={encoder_type!r}; expected one of tcn|transformer|hybrid")


__all__ = ["HybridEncoder", "TCNEncoder", "TransformerEncoder", "build_encoder"]
