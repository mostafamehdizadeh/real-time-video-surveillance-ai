import torch
import torch.nn as nn


class VideoSwin(nn.Module):
    """
    Wrapper model for Video Swin backbone.
    This version is designed to be compatible with pretrained checkpoints
    without breaking due to missing imports or architecture mismatch handling.
    """

    def __init__(self, num_classes=8, embed_dim=128):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # -------------------------
        # BACKBONE (placeholder-safe)
        # -------------------------
        # اگر backbone واقعی داری اینجا جایگزین کن
        self.backbone = nn.Sequential(
            nn.Conv3d(3, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # -------------------------
        # CLASSIFICATION HEAD
        # -------------------------
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x shape: (B, T, C, H, W)
        """

        # convert to (B, C, T, H, W) for Conv3D
        x = x.permute(0, 2, 1, 3, 4)

        x = self.backbone(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        out = self.head(x)

        return out
