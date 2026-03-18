def __getattr__(name):
    """Lazy imports to avoid loading training-only dependencies during inference."""
    if name == "CFM":
        from capspeech.nar.model.cfm import CFM
        return CFM
    elif name == "UNetT":
        from capspeech.nar.model.backbones.unett import UNetT
        return UNetT
    elif name == "DiT":
        from capspeech.nar.model.backbones.dit import DiT
        return DiT
    elif name == "Trainer":
        from capspeech.nar.model.trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
