from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.05) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        confidence = 1.0 - self.smoothing
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (confidence * nll + self.smoothing * smooth_loss).mean()

