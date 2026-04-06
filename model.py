import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard Residual Block with two 3x3 convolutions and a shortcut connection.
    Used throughout the UNet to improve gradient flow and feature extraction.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Shortcut connection to handle dimension changes
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)  # Residual connection
        return self.relu(out)

class AttentionBlock(nn.Module):
    """
    Attention Gate mechanism for UNet.
    Weights the skip connection features (x) using the gating signal from the coarser scale (g).
    focusing on the target structures while suppressing irrelevant background signals.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # Gating signal branch
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Skip connection feature branch
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Attention coefficient calculation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Multiplicative attention weighting

class UNet(nn.Module):
    """
    Attention UNet architecture with Residual Blocks.
    Designed for multi-class medical image segmentation.
    """
    def __init__(self, in_channels=3, out_channels=6, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attentions = nn.ModuleList()

        # Encoder (Downsampling): Residual Blocks followed by MaxPool
        for feature in features:
            self.downs.append(ResidualBlock(in_channels, feature))
            in_channels = feature

        # Decoder (Upsampling): Transposed Convolution, Attention Gate, then Residual Block
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            # Attention block uses signals from current upsampled feature and the skip connection
            self.attentions.append(AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2))
            self.ups.append(ResidualBlock(feature * 2, feature))

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottom of the U
        x = self.bottleneck(x)
        
        # Reverse skip connections for matching in decoder
        skip_connections = skip_connections[::-1]

        # Decoder pass
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]
            
            # Apply Attention to focus the skip connections
            attention_idx = idx // 2
            skip_connection = self.attentions[attention_idx](x, skip_connection)

            # Interpolate if shapes don't match due to padding/stride
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate features and process through Residual Block
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

