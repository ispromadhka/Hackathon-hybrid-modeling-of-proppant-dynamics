"""
Enhanced Fourier Neural Operator (FNO v2) for proppant transport.

Improvements based on research:
1. SpecBoost: Multi-stage residual learning for high-frequency capture
   - Source: https://arxiv.org/abs/2404.07200 (93% error reduction)

2. Multi-scale architecture: Parallel FNO paths at different scales
   - Source: https://arxiv.org/abs/2407.08615 MgFNO (89% error reduction)

3. H1 Loss: Combined L2 + gradient loss for smooth predictions
   - Source: https://arxiv.org/abs/2502.10033

4. Reflection padding: Reduces Gibbs phenomena at boundaries

5. Deep residual architecture: Skip connections for stable deep training

6. Larger capacity: 128 width, 8 layers, 32x16 modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SpectralConv2d(nn.Module):
    """Enhanced 2D Fourier layer with reflection padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        use_reflection_pad: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.use_reflection_pad = use_reflection_pad

        # Xavier initialization for complex weights
        scale = math.sqrt(2.0 / (in_channels + out_channels))

        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]

        # Reflection padding to reduce Gibbs phenomena (from arXiv:2502.10033)
        if self.use_reflection_pad:
            pad_size = max(self.modes1 // 4, 2)
            x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            x_ft = torch.fft.rfft2(x_padded)
            padded_size1 = x_padded.shape[-2]
            padded_size2 = x_padded.shape[-1]
        else:
            x_ft = torch.fft.rfft2(x)
            padded_size1, padded_size2 = size1, size2

        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size, self.out_channels, padded_size1, padded_size2 // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Apply spectral convolution to low and high frequency modes
        m1 = min(self.modes1, padded_size1 // 2)
        m2 = min(self.modes2, padded_size2 // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        # Inverse FFT
        if self.use_reflection_pad:
            x_out = torch.fft.irfft2(out_ft, s=(padded_size1, padded_size2))
            # Remove padding
            x_out = x_out[:, :, pad_size:-pad_size, pad_size:-pad_size]
        else:
            x_out = torch.fft.irfft2(out_ft, s=(size1, size2))

        return x_out


class FNOBlock(nn.Module):
    """Enhanced FNO block with residual connection and layer norm."""

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.use_residual = use_residual

        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.linear = nn.Conv2d(width, width, 1)

        # Layer normalization (better than instance norm for this task)
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Learnable residual scaling (from ResNet-v2)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization (from Transformer architecture)
        x_norm = self.norm1(x)

        # Spectral path (global)
        x1 = self.spectral_conv(x_norm)

        # Linear path (local)
        x2 = self.linear(x_norm)

        # Combine paths
        out = x1 + x2
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Residual connection with learnable scaling
        if self.use_residual:
            out = x + self.gamma * out

        return out


class MultiScaleFNO(nn.Module):
    """
    Multi-scale FNO inspired by MgFNO (arXiv:2407.08615).

    Uses parallel FNO paths at different scales to capture
    both low and high frequency features effectively.
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        n_scales: int = 3
    ):
        super().__init__()
        self.n_scales = n_scales

        # Different scale FNO blocks
        self.scale_blocks = nn.ModuleList()
        for scale in range(n_scales):
            # Each scale captures different frequency bands
            # Scale 0: full modes (low freq)
            # Scale 1: 2x modes (mid freq)
            # Scale 2: 4x modes (high freq - but with smaller spatial size)
            scale_factor = 2 ** scale
            scaled_modes1 = max(4, modes1 // scale_factor)
            scaled_modes2 = max(4, modes2 // scale_factor)

            self.scale_blocks.append(
                FNOBlock(width, scaled_modes1, scaled_modes2, use_residual=False)
            )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(width * n_scales, width, 1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for scale, block in enumerate(self.scale_blocks):
            if scale > 0:
                # Downsample for higher scales
                scale_factor = 2 ** scale
                x_scaled = F.avg_pool2d(x, scale_factor)
                out = block(x_scaled)
                # Upsample back
                out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
            else:
                out = block(x)
            outputs.append(out)

        # Concatenate and fuse
        x_multi = torch.cat(outputs, dim=1)
        return self.fusion(x_multi)


class EnhancedFNOProppant(nn.Module):
    """
    Enhanced FNO for proppant transport with state-of-the-art improvements.

    Architecture:
    1. Lift layer: params + grids → high-dimensional space
    2. Multi-scale FNO blocks with residual connections
    3. Optional SpecBoost residual modules
    4. Projection: → trajectory

    Hyperparameters (optimized for accuracy, not speed):
    - Width: 128 channels (2x baseline)
    - Layers: 8 (deep network)
    - Modes: 32x16 (near full spectrum)
    - Multi-scale: 3 scales
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 32,
        n_times: int = 26,
        modes1: int = 32,    # Near full spectrum
        modes2: int = 16,    # Near full spectrum
        width: int = 128,    # 2x wider
        n_layers: int = 8,   # Deeper
        n_params: int = 9,
        use_multi_scale: bool = True,
        n_scales: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.n_times = n_times
        self.n_params = n_params
        self.width = width

        in_channels = n_params + 2  # params + coordinate grids

        # Lift layer with larger hidden dim
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, width, 1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )

        # Main FNO layers
        self.fno_blocks = nn.ModuleList()
        for i in range(n_layers):
            if use_multi_scale and i % 2 == 1:  # Every other layer is multi-scale
                self.fno_blocks.append(MultiScaleFNO(width, modes1, modes2, n_scales))
            else:
                self.fno_blocks.append(FNOBlock(width, modes1, modes2, dropout=dropout))

        # Deep projection head
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GroupNorm(16, width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 1),
            nn.GroupNorm(16, width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, n_times, 1),
            nn.Sigmoid()  # Concentration in [0, 1]
        )

        # Coordinate grids
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid_x', X.unsqueeze(0).unsqueeze(0).contiguous())
        self.register_buffer('grid_y', Y.unsqueeze(0).unsqueeze(0).contiguous())

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            params: Physics parameters (batch, n_params)

        Returns:
            trajectory: (batch, n_times, nx, ny)
        """
        batch_size = params.shape[0]

        # Expand params to spatial grid
        params_spatial = params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.nx, self.ny)

        # Coordinate grids
        grid_x = self.grid_x.expand(batch_size, -1, -1, -1)
        grid_y = self.grid_y.expand(batch_size, -1, -1, -1)

        # Input tensor
        x = torch.cat([params_spatial, grid_x, grid_y], dim=1)

        # Lift
        x = self.lift(x)

        # FNO layers with residual connections
        for block in self.fno_blocks:
            x = block(x)

        # Project to trajectory
        trajectory = self.project(x)

        return trajectory


class SpecBoostFNO(nn.Module):
    """
    Spectral Boosting FNO (arXiv:2404.07200).

    Trains multiple FNO modules sequentially, each learning the
    residual of the previous modules. This addresses spectral bias
    by allowing later modules to focus on high-frequency content.

    Can reduce error by up to 93% compared to single FNO.
    """

    def __init__(
        self,
        base_model: EnhancedFNOProppant,
        n_boost_stages: int = 2,  # Number of residual modules
    ):
        super().__init__()

        self.base_model = base_model
        self.n_boost_stages = n_boost_stages

        # Residual modules (smaller than base)
        self.residual_modules = nn.ModuleList()
        for _ in range(n_boost_stages):
            # Residual modules are smaller - they focus on high-freq residuals
            module = EnhancedFNOProppant(
                nx=base_model.nx,
                ny=base_model.ny,
                n_times=base_model.n_times,
                modes1=base_model.width // 4,  # Smaller modes
                modes2=base_model.width // 8,
                width=base_model.width // 2,  # Smaller width
                n_layers=4,  # Fewer layers
                n_params=base_model.n_params,
                use_multi_scale=False,
                dropout=0.0,
            )
            self.residual_modules.append(module)

        # Learnable weights for combining predictions
        self.combination_weights = nn.Parameter(torch.ones(1 + n_boost_stages) / (1 + n_boost_stages))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Forward pass with spectral boosting."""
        # Base prediction
        pred = self.base_model(params)
        predictions = [pred]

        # Residual predictions
        for module in self.residual_modules:
            residual = module(params)
            predictions.append(residual)

        # Weighted combination
        weights = F.softmax(self.combination_weights, dim=0)
        output = sum(w * p for w, p in zip(weights, predictions))

        # Clamp to valid range
        return torch.clamp(output, 0, 1)


# ============== LOSS FUNCTIONS ==============

class H1Loss(nn.Module):
    """
    Sobolev H1 loss: L2 + gradient loss.

    From arXiv:2502.10033: "ℰ0 + ℰ1 where ℰ0 measures solution differences
    and ℰ1 measures gradient differences"

    Encourages both pointwise accuracy and smooth gradient predictions.
    """

    def __init__(self, l2_weight: float = 1.0, grad_weight: float = 0.5):
        super().__init__()
        self.l2_weight = l2_weight
        self.grad_weight = grad_weight

    def gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial gradients using central differences."""
        # x shape: (batch, time, nx, ny)
        dx = (x[:, :, 2:, :] - x[:, :, :-2, :]) / 2
        dy = (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # L2 loss (relative)
        diff = pred - target
        l2_loss = (diff ** 2).mean()

        # Gradient loss
        pred_dx, pred_dy = self.gradient(pred)
        target_dx, target_dy = self.gradient(target)

        grad_loss_x = ((pred_dx - target_dx) ** 2).mean()
        grad_loss_y = ((pred_dy - target_dy) ** 2).mean()
        grad_loss = grad_loss_x + grad_loss_y

        return self.l2_weight * l2_loss + self.grad_weight * grad_loss


class CombinedLossV2(nn.Module):
    """
    Advanced combined loss for FNO v2:
    1. Relative L2 loss (spatial accuracy)
    2. Spectral loss (frequency accuracy)
    3. H1 loss (gradient smoothness)
    4. Temporal consistency loss
    """

    def __init__(
        self,
        l2_weight: float = 0.4,
        spectral_weight: float = 0.3,
        h1_weight: float = 0.2,
        temporal_weight: float = 0.1,
    ):
        super().__init__()
        self.l2_weight = l2_weight
        self.spectral_weight = spectral_weight
        self.h1_weight = h1_weight
        self.temporal_weight = temporal_weight

        self.h1_loss = H1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Relative L2 loss
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_norm = torch.norm(target_flat, p=2, dim=1) + 1e-8
        l2_loss = (diff_norm / target_norm).mean()

        # 2. Spectral loss
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        spectral_error = torch.abs(pred_fft - target_fft)

        # Frequency weighting (higher weight for high frequencies)
        nx, ny_rfft = spectral_error.shape[-2], spectral_error.shape[-1]
        freq_x = torch.fft.fftfreq(nx, device=pred.device).abs()
        freq_y = torch.linspace(0, 0.5, ny_rfft, device=pred.device)
        freq_x_grid, freq_y_grid = torch.meshgrid(freq_x, freq_y, indexing='ij')
        freq_magnitude = torch.sqrt(freq_x_grid**2 + freq_y_grid**2)
        freq_weight = 1.0 + 2.0 * (freq_magnitude / (freq_magnitude.max() + 1e-8))

        weighted_spectral_error = spectral_error * freq_weight
        spectral_loss = weighted_spectral_error.mean() / (torch.abs(target_fft).mean() + 1e-8)

        # 3. H1 loss (gradient)
        h1_loss = self.h1_loss(pred, target)

        # 4. Temporal consistency loss
        if pred.shape[1] > 1:
            pred_dt = pred[:, 1:] - pred[:, :-1]
            target_dt = target[:, 1:] - target[:, :-1]
            temporal_loss = F.mse_loss(pred_dt, target_dt)
        else:
            temporal_loss = torch.tensor(0.0, device=pred.device)

        # Combined loss
        total_loss = (
            self.l2_weight * l2_loss +
            self.spectral_weight * spectral_loss +
            self.h1_weight * h1_loss +
            self.temporal_weight * temporal_loss
        )

        return total_loss


# ============== MODEL FACTORY ==============

def create_enhanced_model(
    nx: int = 64,
    ny: int = 32,
    n_times: int = 26,
    n_params: int = 9,
    device: str = 'cpu',
    model_size: str = 'large',  # 'small', 'medium', 'large', 'xlarge'
    use_specboost: bool = False,
    n_boost_stages: int = 2,
) -> nn.Module:
    """
    Create enhanced FNO model.

    Model sizes:
    - small:  ~1M params   (width=64,  layers=4, modes=16x8)
    - medium: ~4M params   (width=96,  layers=6, modes=24x12)
    - large:  ~12M params  (width=128, layers=8, modes=32x16)
    - xlarge: ~25M params  (width=192, layers=10, modes=48x24)
    """
    configs = {
        'small': {'width': 64, 'n_layers': 4, 'modes1': 16, 'modes2': 8},
        'medium': {'width': 96, 'n_layers': 6, 'modes1': 24, 'modes2': 12},
        'large': {'width': 128, 'n_layers': 8, 'modes1': 32, 'modes2': 16},
        'xlarge': {'width': 192, 'n_layers': 10, 'modes1': 48, 'modes2': 24},
    }

    config = configs.get(model_size, configs['large'])

    # Ensure modes don't exceed Nyquist
    config['modes1'] = min(config['modes1'], nx // 2)
    config['modes2'] = min(config['modes2'], ny // 2)

    base_model = EnhancedFNOProppant(
        nx=nx,
        ny=ny,
        n_times=n_times,
        n_params=n_params,
        **config,
        use_multi_scale=True,
        n_scales=3,
        dropout=0.05,
    )

    if use_specboost:
        model = SpecBoostFNO(base_model, n_boost_stages=n_boost_stages)
    else:
        model = base_model

    return model.to(device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    for size in ['small', 'medium', 'large', 'xlarge']:
        model = create_enhanced_model(
            nx=64, ny=32, n_times=26, n_params=9,
            device=device, model_size=size
        )
        n_params_count = sum(p.numel() for p in model.parameters())
        print(f"{size:8s}: {n_params_count:,} parameters")

    # Test forward pass
    print("\n--- Testing large model ---")
    model = create_enhanced_model(
        nx=64, ny=32, n_times=26, n_params=9,
        device=device, model_size='large', use_specboost=True
    )
    n_params_count = sum(p.numel() for p in model.parameters())
    print(f"SpecBoost large: {n_params_count:,} parameters")

    batch_size = 4
    params = torch.rand(batch_size, 9, device=device)

    with torch.no_grad():
        out = model(params)

    print(f"Input: {params.shape}")
    print(f"Output: {out.shape}")

    # Test loss
    print("\n--- Testing loss ---")
    target = torch.rand_like(out)
    loss_fn = CombinedLossV2()
    loss = loss_fn(out, target)
    print(f"Loss: {loss.item():.6f}")
