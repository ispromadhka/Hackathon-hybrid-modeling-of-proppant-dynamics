"""
SuperB-FNO: Multi-scale Fourier Neural Operator for proppant transport.

Улучшенная архитектура с антишумовыми механизмами:
- Низкочастотная фильтрация (подавление высокочастотного шума)
- Гауссово сглаживание выходных данных
- Спектральное сглаживание для плавных переходов
- Temporal smoothing для временной согласованности
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralConv2d(nn.Module):
    """2D Fourier layer with learnable spectral weights and noise suppression."""

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

        # Low frequency modes only (natural noise suppression)
        m1, m2 = min(self.modes1, size1 // 2), min(self.modes2, size2 // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        return x


class LowPassFilter(nn.Module):
    """Низкочастотный фильтр для подавления шума."""

    def __init__(self, cutoff_ratio: float = 0.3):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size1, size2 = x.shape[-2], x.shape[-1]
        x_ft = torch.fft.rfft2(x)

        # Create low-pass mask
        freq1 = torch.fft.fftfreq(size1, device=x.device)
        freq2 = torch.fft.rfftfreq(size2, device=x.device)
        freq1_grid, freq2_grid = torch.meshgrid(freq1, freq2, indexing='ij')

        # Smooth cutoff using Gaussian
        radius = torch.sqrt(freq1_grid**2 + freq2_grid**2)
        mask = torch.exp(-0.5 * (radius / self.cutoff_ratio) ** 4)

        x_ft = x_ft * mask.unsqueeze(0).unsqueeze(0)
        return torch.fft.irfft2(x_ft, s=(size1, size2))


class GaussianSmooth(nn.Module):
    """Гауссово сглаживание для подавления шума."""

    def __init__(self, channels: int, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = g / g.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        # Создаем независимый tensor для каждого канала
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

        self.register_buffer('kernel', kernel_2d)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)


class FNOBlock(nn.Module):
    """FNO block with spectral conv + local conv + residual + smoothing."""

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
        return x + residual


class TemporalSmooth(nn.Module):
    """Сглаживание по временной оси для плавных переходов."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.5):
        super().__init__()
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        kernel = torch.exp(-coords**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_times, nx, ny)
        batch, n_times, nx, ny = x.shape

        # Reshape for 1D convolution along time axis
        x = x.permute(0, 2, 3, 1).reshape(-1, 1, n_times)  # (batch*nx*ny, 1, n_times)

        # Apply temporal smoothing
        kernel = self.kernel.view(1, 1, -1)
        x = F.conv1d(x, kernel, padding=self.padding)

        # Reshape back
        x = x.reshape(batch, nx, ny, n_times).permute(0, 3, 1, 2)
        return x


class SuperBFNO(nn.Module):
    """
    SuperB-FNO с антишумовыми механизмами для предсказания транспорта пропанта.

    Архитектура:
    - Спектральные свёртки с ограничением по частоте
    - Низкочастотная фильтрация
    - Гауссово сглаживание
    - Временное сглаживание для плавных переходов
    """

    def __init__(
        self,
        nx: int = 100,
        ny: int = 100,
        n_times: int = 201,
        width: int = 64,
        n_layers: int = 4,
        n_params: int = 7,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.n_times = n_times
        self.n_params = n_params
        self.width = width

        # Input: params + coordinate grids
        in_channels = n_params + 2

        # Encoder with smooth activation
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, 1),
            nn.GELU(),
            nn.GroupNorm(8, width),
            nn.Conv2d(width, width, 1),
        )

        # FNO layers with decreasing mode counts (progressive frequency reduction)
        self.fno_layers = nn.ModuleList()
        modes_schedule = [
            (16, 16),  # Start with medium frequencies
            (12, 12),  # Reduce
            (10, 10),  # Further reduce
            (8, 8),    # Low frequencies only (smoothest)
        ]

        for i in range(n_layers):
            m1, m2 = modes_schedule[min(i, len(modes_schedule) - 1)]
            self.fno_layers.append(
                FNOBlock(width, m1, m2, dropout=dropout)
            )

        # Low-pass filter for additional noise suppression
        self.lowpass = LowPassFilter(cutoff_ratio=0.25)

        # Parameter injection
        self.param_inject = nn.Sequential(
            nn.Linear(n_params, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

        # Output projection with smooth normalization
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.GroupNorm(8, width * 2),
            nn.Conv2d(width * 2, width, 1),
            nn.GELU(),
            nn.GroupNorm(8, width),
            nn.Conv2d(width, n_times, 1),
        )

        # Post-processing smoothing
        self.spatial_smooth = GaussianSmooth(n_times, kernel_size=3, sigma=0.8)
        self.temporal_smooth = TemporalSmooth(kernel_size=5, sigma=1.2)

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

        # FNO processing
        for layer in self.fno_layers:
            x = layer(x)

        # Apply low-pass filter
        x = self.lowpass(x)

        # Parameter conditioning
        param_embed = self.param_inject(params)
        x = x + param_embed.unsqueeze(-1).unsqueeze(-1)

        # Project to trajectory
        trajectory = self.project(x)

        # Spatial smoothing
        trajectory = self.spatial_smooth(trajectory)

        # Temporal smoothing for smooth time evolution
        trajectory = self.temporal_smooth(trajectory)

        # Smooth clamp using sigmoid-based soft clipping
        # This avoids hard edges from torch.clamp
        trajectory = torch.sigmoid(trajectory * 6 - 3)  # Maps roughly [-0.5, 1.5] to [0, 1]

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
        dropout=0.05,
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
