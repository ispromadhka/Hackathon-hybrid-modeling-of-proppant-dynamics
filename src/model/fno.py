"""
Fourier Neural Operator (FNO) for 2D time-dependent PDEs.

Architecture:
- Input: Initial condition c(x,y,0) + velocity field u(x,y) + parameters
- Output: Full trajectory c(x,y,t) for all time steps

Based on: "Fourier Neural Operator for Parametric Partial Differential Equations"
Li et al., 2020 (https://arxiv.org/abs/2010.08895)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: performs spectral convolution in Fourier space.

    Learns a function in frequency domain that acts as a kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for Fourier modes
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space."""
        # input: (batch, in_channel, x, y)
        # weights: (in_channel, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, nx, ny)

        Returns:
            Output tensor (batch, channels, nx, ny)
        """
        batchsize = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]

        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, size1, size2 // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Lower frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )

        # Higher frequencies (symmetric part)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(size1, size2))

        return x


class FNOBlock(nn.Module):
    """Single FNO block: spectral conv + linear + activation."""

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


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D problems.

    Input channels:
    - c0: Initial concentration (1 channel)
    - u, v: Velocity field (2 channels)
    - x, y: Coordinate grids (2 channels) - for spatial awareness
    Total: 5 input channels

    Output: Full trajectory c(x,y,t) for n_times timesteps
    """

    def __init__(
        self,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 64,
        n_layers: int = 4,
        n_times: int = 40,
        in_channels: int = 5,
    ):
        """
        Args:
            modes1: Fourier modes in x
            modes2: Fourier modes in y
            width: Hidden channel width
            n_layers: Number of FNO layers
            n_times: Number of output time steps
            in_channels: Number of input channels
        """
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_times = n_times

        # Lift input to hidden dimension
        self.lift = nn.Conv2d(in_channels, width, 1)

        # FNO layers
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes1, modes2) for _ in range(n_layers)
        ])

        # Project to output: full trajectory
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, n_times, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, nx, ny)
               Channels: [c0, u, v, x_grid, y_grid]

        Returns:
            out: Output trajectory (batch, n_times, nx, ny)
        """
        # Lift
        x = self.lift(x)

        # FNO layers
        for block in self.fno_blocks:
            x = block(x)

        # Project to trajectory
        out = self.project(x)

        return out


class FNOTrajectoryModel(nn.Module):
    """
    Wrapper that handles input preparation and normalization.
    """

    def __init__(
        self,
        nx: int = 128,
        ny: int = 64,
        n_times: int = 40,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 64,
        n_layers: int = 4
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.n_times = n_times

        self.fno = FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            n_layers=n_layers,
            n_times=n_times,
            in_channels=5
        )

        # Create coordinate grids (registered as buffers)
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', X.unsqueeze(0).unsqueeze(0))  # (1, 1, nx, ny)
        self.register_buffer('grid_y', Y.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        c0: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            c0: Initial concentration (batch, nx, ny) or (batch, 1, nx, ny)
            u: x-velocity field (batch, nx, ny) or (batch, 1, nx, ny)
            v: y-velocity field (batch, nx, ny) or (batch, 1, nx, ny)

        Returns:
            trajectory: (batch, n_times, nx, ny)
        """
        batch_size = c0.shape[0]

        # Ensure 4D tensors
        if c0.dim() == 3:
            c0 = c0.unsqueeze(1)
        if u.dim() == 3:
            u = u.unsqueeze(1)
        if v.dim() == 3:
            v = v.unsqueeze(1)

        # Expand grids to batch size
        grid_x = self.grid_x.expand(batch_size, -1, -1, -1)
        grid_y = self.grid_y.expand(batch_size, -1, -1, -1)

        # Concatenate inputs: (batch, 5, nx, ny)
        x = torch.cat([c0, u, v, grid_x, grid_y], dim=1)

        # Forward through FNO
        trajectory = self.fno(x)

        return trajectory


def create_model(
    nx: int = 128,
    ny: int = 64,
    n_times: int = 40,
    device: str = 'cpu'
) -> FNOTrajectoryModel:
    """Factory function to create the model."""
    model = FNOTrajectoryModel(
        nx=nx,
        ny=ny,
        n_times=n_times,
        modes1=min(16, nx // 4),
        modes2=min(16, ny // 4),
        width=64,
        n_layers=4
    )
    return model.to(device)


if __name__ == '__main__':
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = create_model(nx=128, ny=64, n_times=40, device=device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    batch_size = 4
    c0 = torch.randn(batch_size, 128, 64, device=device)
    u = torch.randn(batch_size, 128, 64, device=device)
    v = torch.zeros(batch_size, 128, 64, device=device)

    with torch.no_grad():
        out = model(c0, u, v)

    print(f"Input shape: {c0.shape}")
    print(f"Output shape: {out.shape}")
