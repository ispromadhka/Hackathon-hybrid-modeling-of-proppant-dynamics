"""
Fourier Neural Operator (FNO) for proppant transport.

Input: Physics parameters [c_inlet, Q_inlet, g, mu0, r_particle, inlet_fraction, rk_stages, lim_type, injection_mode]
Output: Full trajectory c(x,y,t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralConv2d(nn.Module):
    """2D Fourier layer: spectral convolution in Fourier space."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
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

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        return x


class FNOBlock(nn.Module):
    """Single FNO block."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.linear = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.linear(x)
        x = x1 + x2
        x = self.norm(x)
        x = F.gelu(x)
        return x


class FNOProppant(nn.Module):
    """
    FNO for proppant transport prediction.

    Input: Physics parameters encoded as spatial fields + coordinate grids
    Output: Full concentration trajectory c(x,y,t)
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 32,
        n_times: int = 26,
        modes1: int = 12,
        modes2: int = 8,
        width: int = 48,
        n_layers: int = 4,
        n_params: int = 9,  # [c_inlet, Q_inlet, g, mu0, r_particle, inlet_fraction, rk_stages, lim_type, injection_mode]
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.n_times = n_times
        self.n_params = n_params

        # Input: params (n_params) + grids (2) = n_params + 2
        in_channels = n_params + 2

        self.lift = nn.Conv2d(in_channels, width, 1)

        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])

        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, n_times, 1),
            nn.Sigmoid()  # Concentration in [0, 1]
        )

        # Coordinate grids
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', X.unsqueeze(0).unsqueeze(0))
        self.register_buffer('grid_y', Y.unsqueeze(0).unsqueeze(0))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: Physics parameters (batch, n_params)
                    [c_inlet, Q_inlet, g, mu0, r_particle, inlet_fraction, rk_stages, lim_type, injection_mode]

        Returns:
            trajectory: (batch, n_times, nx, ny)
        """
        batch_size = params.shape[0]

        # Expand params to spatial grid: (batch, n_params, nx, ny)
        params_spatial = params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.nx, self.ny)

        # Coordinate grids
        grid_x = self.grid_x.expand(batch_size, -1, -1, -1)
        grid_y = self.grid_y.expand(batch_size, -1, -1, -1)

        # Concatenate: (batch, n_params + 2, nx, ny)
        x = torch.cat([params_spatial, grid_x, grid_y], dim=1)

        # Lift
        x = self.lift(x)

        # FNO layers
        for block in self.fno_blocks:
            x = block(x)

        # Project to trajectory
        trajectory = self.project(x)

        return trajectory


def create_model(
    nx: int = 64,
    ny: int = 32,
    n_times: int = 26,
    n_params: int = 9,
    device: str = 'cpu'
) -> FNOProppant:
    """Create FNO model for proppant transport."""
    model = FNOProppant(
        nx=nx,
        ny=ny,
        n_times=n_times,
        modes1=min(12, nx // 4),
        modes2=min(8, ny // 4),
        width=48,
        n_layers=4,
        n_params=n_params
    )
    return model.to(device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = create_model(nx=64, ny=32, n_times=26, n_params=9, device=device)

    n_params_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params_count:,}")

    # Test forward
    batch_size = 4
    params = torch.rand(batch_size, 9, device=device)

    with torch.no_grad():
        out = model(params)

    print(f"Input: {params.shape}")
    print(f"Output: {out.shape}")
