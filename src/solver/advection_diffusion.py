"""
Numerical solver for 2D Advection-Diffusion equation.

∂c/∂t + u·∇c = D·∇²c

Boundary conditions:
- Left: Dirichlet (inlet)
- Right: Neumann (outflow, ∂c/∂x = 0)
- Top/Bottom: No-flux walls (∂c/∂y = 0)
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class SimulationParams:
    """Parameters for the simulation."""
    Lx: float = 2.0          # Domain length in x [m]
    Ly: float = 1.0          # Domain length in y [m]
    nx: int = 128            # Grid points in x
    ny: int = 64             # Grid points in y
    T: float = 4.0           # Total simulation time [s]
    dt: float = 0.001        # Time step [s]
    D: float = 0.01          # Diffusion coefficient [m²/s]
    U_max: float = 1.0       # Maximum velocity [m/s]
    save_every: int = 10     # Save every N steps


class AdvectionDiffusionSolver:
    """
    Solves 2D advection-diffusion equation using finite differences.

    Uses:
    - Upwind scheme for advection (stable)
    - Central differences for diffusion
    - Explicit Euler time stepping
    """

    def __init__(self, params: SimulationParams):
        self.params = params
        self.dx = params.Lx / (params.nx - 1)
        self.dy = params.Ly / (params.ny - 1)

        # Create grid
        self.x = np.linspace(0, params.Lx, params.nx)
        self.y = np.linspace(0, params.Ly, params.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Initialize velocity field (Poiseuille profile)
        self._init_velocity()

        # Check CFL condition
        self._check_stability()

    def _init_velocity(self):
        """Initialize Poiseuille velocity profile."""
        # Parabolic profile: u(y) = U_max * 4 * y * (Ly - y) / Ly²
        # Maximum at center (y = Ly/2), zero at walls
        y_norm = self.Y / self.params.Ly
        self.u = self.params.U_max * 4 * y_norm * (1 - y_norm)
        self.v = np.zeros_like(self.u)  # No vertical velocity

    def _check_stability(self):
        """Check CFL and diffusion stability conditions."""
        dt = self.params.dt
        dx, dy = self.dx, self.dy
        D = self.params.D
        U = self.params.U_max

        # CFL for advection
        cfl_x = U * dt / dx
        cfl_y = U * dt / dy

        # Diffusion stability
        diff_stability = D * dt * (1/dx**2 + 1/dy**2)

        if cfl_x > 1 or cfl_y > 1:
            print(f"Warning: CFL condition violated! CFL_x={cfl_x:.3f}, CFL_y={cfl_y:.3f}")

        if diff_stability > 0.5:
            print(f"Warning: Diffusion stability violated! Factor={diff_stability:.3f}")

    def _apply_boundary_conditions(self, c: np.ndarray, c_inlet: float = 0.0):
        """Apply boundary conditions."""
        # Left: Dirichlet (inlet) - will be set by initial condition or inlet function
        c[0, :] = c_inlet

        # Right: Neumann (outflow, ∂c/∂x = 0)
        c[-1, :] = c[-2, :]

        # Top/Bottom: No-flux (∂c/∂y = 0)
        c[:, 0] = c[:, 1]
        c[:, -1] = c[:, -2]

        return c

    def _advection_step(self, c: np.ndarray) -> np.ndarray:
        """Compute advection term using upwind scheme."""
        dc = np.zeros_like(c)

        # Upwind for x-direction (u > 0 everywhere in Poiseuille)
        # dc/dx ≈ (c[i] - c[i-1]) / dx for u > 0
        dc[1:, :] -= self.u[1:, :] * (c[1:, :] - c[:-1, :]) / self.dx

        # v = 0, so no y-advection needed

        return dc

    def _diffusion_step(self, c: np.ndarray) -> np.ndarray:
        """Compute diffusion term using central differences."""
        dc = np.zeros_like(c)
        D = self.params.D

        # ∂²c/∂x²
        dc[1:-1, :] += D * (c[2:, :] - 2*c[1:-1, :] + c[:-2, :]) / self.dx**2

        # ∂²c/∂y²
        dc[:, 1:-1] += D * (c[:, 2:] - 2*c[:, 1:-1] + c[:, :-2]) / self.dy**2

        return dc

    def solve(
        self,
        c0: np.ndarray,
        inlet_func: Optional[Callable[[float], float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the advection-diffusion equation.

        Args:
            c0: Initial concentration field (nx, ny)
            inlet_func: Optional function f(t) -> inlet_concentration

        Returns:
            times: Array of saved time points
            concentrations: Array of concentration fields (n_times, nx, ny)
        """
        p = self.params
        n_steps = int(p.T / p.dt)
        n_saves = n_steps // p.save_every + 1

        # Initialize storage
        times = np.zeros(n_saves)
        concentrations = np.zeros((n_saves, p.nx, p.ny))

        # Initial condition
        c = c0.copy()
        c = self._apply_boundary_conditions(c, c_inlet=0.0)

        concentrations[0] = c.copy()
        times[0] = 0.0

        save_idx = 1

        for step in range(1, n_steps + 1):
            t = step * p.dt

            # Compute RHS
            dc_adv = self._advection_step(c)
            dc_diff = self._diffusion_step(c)

            # Euler step
            c = c + p.dt * (dc_adv + dc_diff)

            # Apply BCs
            inlet_val = inlet_func(t) if inlet_func else 0.0
            c = self._apply_boundary_conditions(c, c_inlet=inlet_val)

            # Ensure non-negative concentration
            c = np.maximum(c, 0.0)

            # Save
            if step % p.save_every == 0:
                concentrations[save_idx] = c.copy()
                times[save_idx] = t
                save_idx += 1

        return times, concentrations


def create_initial_condition(
    X: np.ndarray,
    Y: np.ndarray,
    ic_type: str = 'gaussian',
    **kwargs
) -> np.ndarray:
    """
    Create initial concentration field.

    Args:
        X, Y: Meshgrid arrays
        ic_type: Type of initial condition
            - 'gaussian': Gaussian blob
            - 'step': Step function
            - 'random': Random field
        **kwargs: Parameters for specific IC types

    Returns:
        c0: Initial concentration field
    """
    if ic_type == 'gaussian':
        x0 = kwargs.get('x0', 0.3)
        y0 = kwargs.get('y0', 0.5)
        sigma_x = kwargs.get('sigma_x', 0.1)
        sigma_y = kwargs.get('sigma_y', 0.15)
        amplitude = kwargs.get('amplitude', 0.7)

        c0 = amplitude * np.exp(
            -((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))
        )

    elif ic_type == 'step':
        x_thresh = kwargs.get('x_thresh', 0.5)
        amplitude = kwargs.get('amplitude', 0.5)
        c0 = np.where(X < x_thresh, amplitude, 0.0)

    elif ic_type == 'random':
        np.random.seed(kwargs.get('seed', None))
        amplitude = kwargs.get('amplitude', 0.5)
        c0 = amplitude * np.random.rand(*X.shape)
        # Smooth it a bit
        from scipy.ndimage import gaussian_filter
        c0 = gaussian_filter(c0, sigma=2)

    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    return c0


def generate_training_sample(
    params: SimulationParams,
    ic_params: dict,
    inlet_func: Optional[Callable] = None
) -> dict:
    """
    Generate one training sample.

    Returns:
        dict with keys:
            - 'params': simulation parameters
            - 'ic_params': initial condition parameters
            - 'times': time array
            - 'concentrations': concentration fields over time
            - 'velocity_u': x-velocity field
            - 'velocity_v': y-velocity field
    """
    solver = AdvectionDiffusionSolver(params)

    c0 = create_initial_condition(
        solver.X, solver.Y, **ic_params
    )

    times, concentrations = solver.solve(c0, inlet_func)

    return {
        'params': params,
        'ic_params': ic_params,
        'times': times,
        'concentrations': concentrations,
        'velocity_u': solver.u,
        'velocity_v': solver.v,
        'X': solver.X,
        'Y': solver.Y
    }


if __name__ == '__main__':
    # Quick test
    params = SimulationParams(
        nx=128, ny=64,
        T=4.0, dt=0.001,
        D=0.01, U_max=1.0,
        save_every=50
    )

    sample = generate_training_sample(
        params,
        ic_params={'ic_type': 'gaussian', 'x0': 0.3, 'y0': 0.5}
    )

    print(f"Generated {len(sample['times'])} frames")
    print(f"Concentration shape: {sample['concentrations'].shape}")
    print(f"Time range: {sample['times'][0]:.2f} - {sample['times'][-1]:.2f}")
