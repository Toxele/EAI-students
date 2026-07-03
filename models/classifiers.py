from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn
except ImportError:  # pragma: no cover
    nn = None


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class ClassifierFactory:
    @staticmethod
    def create(cfg: dict[str, Any], num_classes: int):
        if nn is None:
            raise ImportError("ClassifierFactory requires torch.")
        name = cfg.get("name", "small_cnn").lower()
        in_channels = cfg.get("in_channels", 3)
        if name == "small_cnn":
            return SmallCNN(num_classes=num_classes, in_channels=in_channels)
        if name.startswith("resnet"):
            try:
                import torchvision.models as models

                weights = "DEFAULT" if cfg.get("pretrained", True) else None
                model_fn = getattr(models, name)
                model = model_fn(weights=weights)
                if in_channels != 3:
                    old_conv = model.conv1
                    new_conv = nn.Conv2d(
                        in_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None,
                    )
                    new_conv.weight.data[:, :3] = old_conv.weight.data
                    if in_channels > 3:
                        extra = old_conv.weight.data.mean(dim=1, keepdim=True)
                        new_conv.weight.data[:, 3:in_channels] = extra.repeat(1, in_channels - 3, 1, 1)
                    model.conv1 = new_conv
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                return model
            except Exception:
                if cfg.get("fallback", "small_cnn") == "small_cnn":
                    return SmallCNN(num_classes=num_classes, in_channels=in_channels)
                raise
        raise ValueError(f"Unknown classifier model: {name}")
