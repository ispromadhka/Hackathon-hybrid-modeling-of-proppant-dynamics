"""
SuperB-FNO: Multi-scale Fourier Neural Operator for proppant transport.

Architecture improvements over vanilla FNO:
- Multi-scale spectral convolutions (different modes at each level)
- U-Net style encoder-decoder with skip connections
- Residual learning for better gradient flow
- Spectral attention for adaptive frequency weighting
- High-Frequency Scaling (HFS) for spectral bias mitigation
- Boundary-aware processing for mass conservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer with learnable spectral weights.

    Includes High-Frequency Scaling (HFS) to mitigate spectral bias:
    - Higher frequencies get amplified scaling during forward pass
    - Helps model learn fine-grained features
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int,
                 use_hfs: bool = True, hfs_alpha: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.use_hfs = use_hfs
        self.hfs_alpha = hfs_alpha

        # Xavier-like initialization for complex weights
        scale = 1 / math.sqrt(in_channels * out_channels)

        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

        # High-frequency scaling weights (learnable)
        if use_hfs:
            self.hf_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def get_frequency_scaling(self, size1: int, rfft_size2: int, orig_size2: int, device: torch.device) -> torch.Tensor:
        """Generate frequency-dependent scaling to amplify high frequencies."""
        # Create frequency grid matching rfft2 output dimensions
        freq1 = torch.fft.fftfreq(size1, device=device)
        freq2 = torch.fft.rfftfreq(orig_size2, device=device)
        freq_grid = torch.sqrt(freq1[:, None]**2 + freq2[None, :]**2)

        # Normalize to [0, 1] and apply scaling
        freq_grid = freq_grid / (freq_grid.max() + 1e-8)

        # HFS: scale = 1 + alpha * normalized_frequency
        scaling = 1.0 + self.hfs_alpha * freq_grid
        return scaling.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]
        orig_dtype = x.dtype

        # cuFFT doesn't support half precision for non-power-of-two sizes
        # Always compute FFT in float32
        x_float = x.float()
        x_ft = torch.fft.rfft2(x_float)

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

        # Apply High-Frequency Scaling
        if self.use_hfs:
            hf_scaling = self.get_frequency_scaling(size1, size2 // 2 + 1, size2, x.device)
            out_ft = out_ft * hf_scaling

        x = torch.fft.irfft2(out_ft, s=(size1, size2))

        # Apply learnable high-frequency boost
        if self.use_hfs:
            x = x * self.hf_scale

        # Convert back to original dtype (for AMP compatibility)
        return x.to(orig_dtype)


class SpectralAttention(nn.Module):
    """
    Learnable attention over frequency components with multi-band weighting.

    Features:
    - Separate weights for low, mid, and high frequency bands
    - Learnable band boundaries
    - Better gradient flow for high-frequency learning
    """

    def __init__(self, channels: int, modes1: int, modes2: int, n_bands: int = 3):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_bands = n_bands

        # Multi-band frequency weights
        self.freq_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, modes1, modes2) * (1.0 / (i + 1)))
            for i in range(n_bands)
        ])

        # Learnable band boundaries (as fractions of max frequency)
        self.band_boundaries = nn.Parameter(
            torch.linspace(0.2, 0.8, n_bands - 1)
        )

        # High-frequency boost (learnable)
        self.hf_boost = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, size1, size2 = x.shape
        orig_dtype = x.dtype

        # cuFFT doesn't support half precision for non-power-of-two sizes
        x_float = x.float()
        x_ft = torch.fft.rfft2(x_float)

        m1, m2 = min(self.modes1, size1 // 2), min(self.modes2, size2 // 2 + 1)

        # Create frequency distance grid
        freq1 = torch.arange(m1, device=x.device).float() / m1
        freq2 = torch.arange(m2, device=x.device).float() / m2
        freq_dist = torch.sqrt(freq1[:, None]**2 + freq2[None, :]**2)
        freq_dist = freq_dist / (freq_dist.max() + 1e-8)

        # Apply multi-band weighting
        combined_mask = torch.zeros(1, channels, m1, m2, device=x.device)
        boundaries = torch.sigmoid(self.band_boundaries)  # Ensure [0, 1]
        boundaries = torch.cat([torch.zeros(1, device=x.device), boundaries, torch.ones(1, device=x.device)])

        for i, weight in enumerate(self.freq_weights):
            low, high = boundaries[i], boundaries[i + 1]
            band_mask = ((freq_dist >= low) & (freq_dist < high)).float()

            # Apply higher boost to higher frequency bands
            boost = 1.0 + self.hf_boost * (i / (self.n_bands - 1))
            combined_mask = combined_mask + F.softplus(weight[:, :, :m1, :m2]) * band_mask * boost

        # Create output tensor (no in-place ops for autograd)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :m1, :m2] = x_ft[:, :, :m1, :m2] * combined_mask
        # Copy remaining frequencies with slight attenuation
        out_ft[:, :, m1:, :] = x_ft[:, :, m1:, :] * 0.1
        out_ft[:, :, :m1, m2:] = x_ft[:, :, :m1, m2:] * 0.1

        result = torch.fft.irfft2(out_ft, s=(size1, size2))
        return result.to(orig_dtype)


class BoundaryMask(nn.Module):
    """
    Learnable boundary mask for mass conservation.

    Creates soft masks at domain boundaries to:
    - Prevent artificial mass loss at edges
    - Enforce no-flux boundary conditions
    - Improve physical consistency
    """

    def __init__(self, nx: int, ny: int, boundary_width: int = 2):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.boundary_width = boundary_width

        # Create base boundary mask (1 in interior, 0 at boundaries)
        mask = torch.ones(1, 1, nx, ny)

        # Top and bottom boundaries (no-flux for proppant)
        for i in range(boundary_width):
            weight = (i + 1) / (boundary_width + 1)
            mask[:, :, :, i] = weight
            mask[:, :, :, -(i + 1)] = weight

        # Learnable boundary correction
        self.base_mask = nn.Parameter(mask, requires_grad=False)
        self.boundary_correction = nn.Parameter(torch.zeros(1, 1, nx, ny))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply boundary mask to output."""
        mask = self.base_mask + torch.tanh(self.boundary_correction) * 0.1
        mask = torch.clamp(mask, 0, 1)
        return x * mask


class ResidualHighFrequencyBlock(nn.Module):
    """
    SpecBoost-inspired residual block for high-frequency learning.

    Instead of learning everything at once, learn:
    1. Low-frequency base prediction
    2. High-frequency residual correction

    This helps overcome spectral bias.
    """

    def __init__(self, width: int, modes_low: tuple, modes_high: tuple, dropout: float = 0.0):
        super().__init__()

        # Low-frequency path
        self.low_freq = SpectralConv2d(width, width, modes_low[0], modes_low[1], use_hfs=False)

        # High-frequency residual path
        self.high_freq = SpectralConv2d(width, width, modes_high[0], modes_high[1], use_hfs=True)

        # Combine
        self.combine = nn.Conv2d(width * 2, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Low-frequency features
        x_low = self.low_freq(x)

        # High-frequency residual
        x_high = self.high_freq(x)

        # Combine both paths
        x_combined = torch.cat([x_low, x_high], dim=1)
        x = self.combine(x_combined)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)

        return x + residual


class FNOBlock(nn.Module):
    """FNO block with spectral conv + local conv + residual + HFS."""

    def __init__(self, width: int, modes1: int, modes2: int, dropout: float = 0.0, use_hfs: bool = True):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2, use_hfs=use_hfs)
        self.local_conv = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Spatial attention for local feature enhancement
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(width, width // 4, 1),
            nn.GELU(),
            nn.Conv2d(width // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.spectral_conv(x)
        x2 = self.local_conv(x)

        # Spatial attention weighting
        attn = self.spatial_attn(x)
        x = x1 + x2 * attn

        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual  # Residual connection


class MultiScaleFNOBlock(nn.Module):
    """
    Multi-scale FNO block with different frequency modes.

    Based on MscaleFNO research - uses progressive frequency bands
    to capture both coarse and fine features simultaneously.
    """

    def __init__(self, width: int, modes_list: list, dropout: float = 0.0):
        super().__init__()
        n_branches = len(modes_list)
        branch_width = width // n_branches

        self.branches = nn.ModuleList([
            SpectralConv2d(width, branch_width, m1, m2, use_hfs=(i >= n_branches // 2))
            for i, (m1, m2) in enumerate(modes_list)
        ])
        self.local_conv = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable scale mixing weights
        self.scale_weights = nn.Parameter(torch.ones(n_branches) / n_branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Multi-scale spectral processing
        spectral_outs = []
        weights = F.softmax(self.scale_weights, dim=0)
        for i, branch in enumerate(self.branches):
            out = branch(x) * weights[i]
            spectral_outs.append(out)

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

    Enhanced multi-scale architecture with:
    - Progressive frequency capture (fine to coarse)
    - High-Frequency Scaling (HFS) for spectral bias mitigation
    - Boundary masks for mass conservation
    - SpecBoost-style residual learning
    - Parameter conditioning at multiple levels
    - Adaptive mass conservation correction
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
        use_boundary_mask: bool = True,
        use_mass_conservation: bool = True,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.n_times = n_times
        self.n_params = n_params
        self.width = width
        self.use_boundary_mask = use_boundary_mask
        self.use_mass_conservation = use_mass_conservation

        # Input: params + coordinate grids
        in_channels = n_params + 2

        # Encoder with larger initial projection
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, 1),
            nn.GELU(),
            nn.GroupNorm(8, width),
            nn.Conv2d(width, width, 1),
        )

        # Improved modes_schedule based on MscaleFNO research
        # Key insight: overlap frequency bands for smooth transitions
        modes_schedule = [
            (32, 32),  # Very fine details (high frequency)
            (24, 24),  # Fine details
            (16, 16),  # Medium scale
            (12, 12),  # Coarse features
            (8, 8),    # Global patterns
            (6, 6),    # Very global
        ]

        # Main FNO layers
        self.fno_layers = nn.ModuleList()
        for i in range(n_layers):
            m1, m2 = modes_schedule[min(i, len(modes_schedule) - 1)]
            # Use HFS in later layers (where we need high-frequency details)
            use_hfs = (i >= n_layers // 2)
            self.fno_layers.append(
                FNOBlock(width, m1, m2, dropout=dropout, use_hfs=use_hfs)
            )

        # SpecBoost-style residual high-frequency block
        self.hf_residual = ResidualHighFrequencyBlock(
            width,
            modes_low=(8, 8),
            modes_high=(24, 24),
            dropout=dropout
        )

        # Multi-scale fusion block
        multiscale_modes = [(24, 24), (16, 16), (8, 8), (4, 4)]
        self.multiscale_block = MultiScaleFNOBlock(width, multiscale_modes, dropout=dropout)

        # Spectral attention for adaptive frequency selection
        self.spectral_attention = SpectralAttention(width, 20, 20, n_bands=4)

        # Parameter injection at multiple points
        self.param_inject_early = nn.Sequential(
            nn.Linear(n_params, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

        self.param_inject_late = nn.Sequential(
            nn.Linear(n_params, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )

        # Time-aware output projection
        self.time_embed = nn.Sequential(
            nn.Linear(n_times, width),
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
            nn.GroupNorm(8, width * 2),
            nn.Conv2d(width * 2, n_times, 1),
        )

        # Boundary mask for mass conservation
        if use_boundary_mask:
            self.boundary_mask = BoundaryMask(nx, ny, boundary_width=2)

        # Mass conservation correction (learnable scaling per timestep)
        if use_mass_conservation:
            self.mass_scale = nn.Parameter(torch.ones(1, n_times, 1, 1))

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

        # Early parameter conditioning
        param_embed_early = self.param_inject_early(params)
        x = x + param_embed_early.unsqueeze(-1).unsqueeze(-1)

        # Main FNO processing
        for i, layer in enumerate(self.fno_layers):
            x = layer(x)

        # High-frequency residual learning (SpecBoost-style)
        x = self.hf_residual(x)

        # Multi-scale fusion
        x = self.multiscale_block(x)

        # Spectral attention
        x = self.spectral_attention(x)

        # Late parameter conditioning
        param_embed_late = self.param_inject_late(params)
        x = x + param_embed_late.unsqueeze(-1).unsqueeze(-1)

        # Project to trajectory
        trajectory = self.project(x)

        # Apply boundary mask
        if self.use_boundary_mask:
            trajectory = self.boundary_mask(trajectory)

        # Apply mass conservation scaling
        if self.use_mass_conservation:
            trajectory = trajectory * F.softplus(self.mass_scale)

        # Clamp output to valid range [0, 1]
        trajectory = torch.clamp(trajectory, 0.0, 1.0)

        return trajectory

    def get_mass_at_timestep(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute total mass at each timestep for conservation loss.

        Args:
            trajectory: (batch, n_times, nx, ny)

        Returns:
            mass: (batch, n_times) - total mass at each timestep
        """
        # Sum over spatial dimensions
        return trajectory.sum(dim=(-2, -1))


def create_model(
    nx: int = 100,
    ny: int = 100,
    n_times: int = 201,
    n_params: int = 7,
    device: str = 'cpu',
    use_boundary_mask: bool = True,
    use_mass_conservation: bool = True,
) -> SuperBFNO:
    """
    Create SuperB-FNO model for proppant transport.

    Enhanced with:
    - High-Frequency Scaling (HFS) for spectral bias mitigation
    - Boundary masks for mass conservation
    - SpecBoost-style residual learning
    - Multi-scale frequency processing
    """
    model = SuperBFNO(
        nx=nx,
        ny=ny,
        n_times=n_times,
        width=64,
        n_layers=4,
        n_params=n_params,
        dropout=0.1,
        use_boundary_mask=use_boundary_mask,
        use_mass_conservation=use_mass_conservation,
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

    # Test mass computation
    mass = model.get_mass_at_timestep(out)
    print(f"Mass per timestep shape: {mass.shape}")
    print(f"Mass range: [{mass.min().item():.2f}, {mass.max().item():.2f}]")
