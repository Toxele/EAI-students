from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover
    smp = None


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a two-layer convolution block used by the local U-Net."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply two convolution, batch norm and ReLU stages."""
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        """Create a compact U-Net for binary segmentation experiments."""
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        """Return one-channel segmentation logits for an input batch."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SegmentationFactory:
    @staticmethod
    def create(cfg: dict):
        """Build a segmentation model from a small JSON-friendly config."""
        name = cfg.get("name", "tiny_unet").lower()
        if name == "tiny_unet":
            return TinyUNet(
                in_channels=cfg.get("in_channels", 3),
                base_channels=cfg.get("base_channels", 32),
            )
        if name in {"smp_unet", "unet"}:
            if smp is None:
                raise ImportError("Install segmentation-models-pytorch to use smp_unet.")
            return smp.Unet(
                encoder_name=cfg.get("encoder_name", "resnet34"),
                encoder_weights=cfg.get("encoder_weights", "imagenet"),
                in_channels=cfg.get("in_channels", 3),
                classes=cfg.get("classes", 1),
                activation=None,
            )
        raise ValueError(f"Unknown segmentation model: {name}")
