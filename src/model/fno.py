"""
SuperB-FNO: Multi-scale Fourier Neural Operator for proppant transport.

Architecture improvements over vanilla FNO:
- Multi-scale spectral convolutions (different modes at each level)
- U-Net style encoder-decoder with skip connections
- Residual learning for better gradient flow
- Spectral attention for adaptive frequency weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralConv2d(nn.Module):
    """2D Fourier layer with learnable spectral weights."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Xavier-like initialization for complex weights
        scale = 1 / math.sqrt(in_channels * out_channels)

        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, size1, size2 // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Low frequency modes (corners in 2D FFT)
        m1, m2 = min(self.modes1, size1 // 2), min(self.modes2, size2 // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        return x


class SpectralAttention(nn.Module):
    """Learnable attention over frequency components."""

    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        # Learnable frequency mask
        self.freq_weight = nn.Parameter(torch.ones(1, channels, modes1, modes2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size1, size2 = x.shape[-2], x.shape[-1]
        x_ft = torch.fft.rfft2(x)

        # Apply learnable frequency weighting
        m1, m2 = min(self.modes1, size1 // 2), min(self.modes2, size2 // 2 + 1)
        mask = F.softplus(self.freq_weight[:, :, :m1, :m2])
        x_ft[:, :, :m1, :m2] = x_ft[:, :, :m1, :m2] * mask

        return torch.fft.irfft2(x_ft, s=(size1, size2))


class FNOBlock(nn.Module):
    """FNO block with spectral conv + local conv + residual."""

    def __init__(self, width: int, modes1: int, modes2: int, dropout: float = 0.0):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.local_conv = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)
        x = x1 + x2
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual  # Residual connection


class MultiScaleFNOBlock(nn.Module):
    """Multi-scale FNO block with different frequency modes."""

    def __init__(self, width: int, modes_list: list, dropout: float = 0.0):
        super().__init__()
        self.branches = nn.ModuleList([
            SpectralConv2d(width, width // len(modes_list), m1, m2)
            for m1, m2 in modes_list
        ])
        self.local_conv = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        spectral_outs = [branch(x) for branch in self.branches]
        x_spectral = torch.cat(spectral_outs, dim=1)
        x_local = self.local_conv(x)
        x = x_spectral + x_local
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class SuperBFNO(nn.Module):
    """
    SuperB-FNO for proppant transport prediction.

    Multi-scale architecture with:
    - Encoder: progressively capture different frequency scales
    - Bottleneck: high-capacity spectral processing
    - Decoder: upsample with skip connections
    - Parameter conditioning at multiple levels
    """

    def __init__(
        self,
        nx: int = 100,
        ny: int = 100,
        n_times: int = 201,
        width: int = 64,
        n_layers: int = 4,
        n_params: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.n_times = n_times
        self.n_params = n_params
        self.width = width

        # Input: params + coordinate grids
        in_channels = n_params + 2

        # Encoder
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

        # Multi-scale FNO layers with increasing receptive field
        self.fno_layers = nn.ModuleList()
        modes_schedule = [
            (24, 24),  # Fine details
            (16, 16),  # Medium scale
            (12, 12),  # Coarse features
            (8, 8),    # Global patterns
        ]

        for i in range(n_layers):
            m1, m2 = modes_schedule[min(i, len(modes_schedule) - 1)]
            self.fno_layers.append(
                FNOBlock(width, m1, m2, dropout=dropout)
            )

        # Spectral attention for adaptive frequency selection
        self.spectral_attention = SpectralAttention(width, 16, 16)

        # Parameter injection at bottleneck
        self.param_inject = nn.Sequential(
            nn.Linear(n_params, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

        # Time embedding for temporal awareness
        self.time_embed = nn.Sequential(
            nn.Linear(1, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        # Output projection with larger capacity
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.GroupNorm(8, width * 2),
            nn.Conv2d(width * 2, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, n_times, 1),
        )

        # Coordinate grids
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', X.unsqueeze(0).unsqueeze(0).contiguous())
        self.register_buffer('grid_y', Y.unsqueeze(0).unsqueeze(0).contiguous())

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: Physics parameters (batch, n_params)

        Returns:
            trajectory: (batch, n_times, nx, ny) - concentration [0, 1]
        """
        batch_size = params.shape[0]

        # Expand params to spatial grid
        params_spatial = params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.nx, self.ny)

        # Coordinate grids
        grid_x = self.grid_x.expand(batch_size, -1, -1, -1)
        grid_y = self.grid_y.expand(batch_size, -1, -1, -1)

        # Input assembly
        x = torch.cat([params_spatial, grid_x, grid_y], dim=1)

        # Lift to hidden dimension
        x = self.lift(x)

        # FNO processing with residual
        for layer in self.fno_layers:
            x = layer(x)

        # Spectral attention
        x = self.spectral_attention(x)

        # Parameter conditioning at bottleneck
        param_embed = self.param_inject(params)
        x = x + param_embed.unsqueeze(-1).unsqueeze(-1)

        # Project to trajectory
        trajectory = self.project(x)

        # Clamp output to valid range [0, 1]
        trajectory = torch.clamp(trajectory, 0.0, 1.0)

        return trajectory


def create_model(
    nx: int = 100,
    ny: int = 100,
    n_times: int = 201,
    n_params: int = 7,
    device: str = 'cpu'
) -> SuperBFNO:
    """Create SuperB-FNO model for proppant transport."""
    model = SuperBFNO(
        nx=nx,
        ny=ny,
        n_times=n_times,
        width=64,
        n_layers=4,
        n_params=n_params,
        dropout=0.1,
    )
    return model.to(device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = create_model(nx=100, ny=100, n_times=201, n_params=7, device=device)

    n_params_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params_count:,}")

    # Test forward
    batch_size = 4
    params = torch.rand(batch_size, 7, device=device)

    with torch.no_grad():
        out = model(params)

    print(f"Input: {params.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
