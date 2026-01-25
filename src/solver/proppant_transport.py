"""
Two-phase proppant transport solver.

Equations:
    ∂(cw)/∂t + ∇·(cwVₚ) = 0              — mass conservation
    Vf = Poiseuille profile               — slot flow
    Vₚ = Vf + Vslip(c)                    — proppant velocity

Closure models:
    μ(c)  — Nolte viscosity
    ρ(c)  — linear mixture density
    Vslip — Richardson-Zaki hindered settling
"""

import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhysicalParams:
    """Physical parameters for proppant transport."""
    # Fluid properties
    rho_f: float = 1000.0       # Fluid density [kg/m³]
    mu_f: float = 0.01          # Fluid viscosity [Pa·s] (10 mPa·s)

    # Proppant properties
    rho_p: float = 2650.0       # Proppant density [kg/m³]
    d_p: float = 0.0004         # Proppant diameter [m] (400 μm)
    c_max: float = 0.65         # Maximum packing fraction

    # Gravity
    g: float = 9.81             # Gravitational acceleration [m/s²]

    # Fracture
    w0: float = 0.005           # Fracture width [m] (5 mm)

    # Flow
    U_max: float = 0.8          # Maximum flow velocity [m/s]


@dataclass
class SimulationParams:
    """Simulation parameters."""
    Lx: float = 2.0             # Domain length x [m]
    Ly: float = 1.0             # Domain length y [m]
    nx: int = 100               # Grid points x
    ny: int = 50                # Grid points y
    T: float = 3.0              # Total time [s]
    dt: float = 0.002           # Time step [s]
    save_every: int = 25        # Save interval
    c_inlet: float = 0.35       # Inlet concentration


@njit(cache=True)
def mixture_density(c: float, rho_f: float, rho_p: float) -> float:
    """Linear mixture density: ρ(c) = ρf(1-c) + ρp·c"""
    return rho_f * (1.0 - c) + rho_p * c


@njit(cache=True)
def nolte_viscosity(c: float, mu_f: float, c_max: float) -> float:
    """
    Nolte viscosity model for proppant slurry:
    μ(c) = μf · (1 + 1.25·c / (1 - c/c_max))
    """
    if c <= 0:
        return mu_f
    if c >= c_max * 0.98:
        c = c_max * 0.98
    return mu_f * (1.0 + 1.25 * c / (1.0 - c / c_max))


@njit(cache=True)
def settling_velocity(c: float, rho_f: float, rho_p: float,
                      mu_f: float, d_p: float, g: float, c_max: float) -> float:
    """
    Richardson-Zaki hindered settling:
    Vslip = V_stokes · (1-c/c_max)^n

    V_stokes = (ρp - ρf) · g · d² / (18·μf)
    n ≈ 4.65 for creeping flow
    """
    if c <= 0:
        # Single particle settling
        V_stokes = (rho_p - rho_f) * g * d_p * d_p / (18.0 * mu_f)
        return -V_stokes
    if c >= c_max * 0.98:
        return 0.0

    # Stokes settling velocity
    V_stokes = (rho_p - rho_f) * g * d_p * d_p / (18.0 * mu_f)

    # Hindered settling (Richardson-Zaki)
    n = 4.65
    hindrance = (1.0 - c / c_max) ** n

    return -V_stokes * hindrance  # Negative = downward


@njit(cache=True)
def compute_slip_velocity_field(c: np.ndarray, phys: tuple) -> np.ndarray:
    """Compute vertical slip velocity field."""
    rho_f, rho_p, mu_f, d_p, g, c_max, U_max = phys
    nx, ny = c.shape
    v_slip = np.zeros_like(c)
    for i in range(nx):
        for j in range(ny):
            v_slip[i, j] = settling_velocity(c[i, j], rho_f, rho_p, mu_f, d_p, g, c_max)
    return v_slip


@njit(cache=True)
def create_poiseuille_velocity(nx: int, ny: int, Ly: float, U_max: float) -> np.ndarray:
    """
    Create Poiseuille velocity profile: u(y) = U_max * 4 * y/Ly * (1 - y/Ly)
    Maximum at center (y = Ly/2), zero at walls.
    """
    u = np.zeros((nx, ny))
    for j in range(ny):
        y_norm = j / (ny - 1)  # 0 to 1
        vel = U_max * 4.0 * y_norm * (1.0 - y_norm)
        for i in range(nx):
            u[i, j] = vel
    return u


@njit(cache=True)
def advection_step(c: np.ndarray, u_p: np.ndarray, v_p: np.ndarray,
                   dx: float, dy: float, dt: float, c_max: float) -> np.ndarray:
    """
    Advection step for concentration using upwind scheme.
    ∂c/∂t + ∇·(c·Vₚ) = 0
    """
    nx, ny = c.shape
    c_new = c.copy()

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Upwind scheme for advection
            # X-direction
            if u_p[i, j] > 0:
                dc_dx = (c[i, j] - c[i-1, j]) / dx
            else:
                dc_dx = (c[i+1, j] - c[i, j]) / dx

            # Y-direction
            if v_p[i, j] > 0:
                dc_dy = (c[i, j] - c[i, j-1]) / dy
            else:
                dc_dy = (c[i, j+1] - c[i, j]) / dy

            # Advection
            advection = u_p[i, j] * dc_dx + v_p[i, j] * dc_dy

            c_new[i, j] = c[i, j] - dt * advection

    # Clamp concentration
    for i in range(nx):
        for j in range(ny):
            if c_new[i, j] < 0:
                c_new[i, j] = 0.0
            if c_new[i, j] > c_max:
                c_new[i, j] = c_max

    return c_new


@njit(cache=True)
def apply_boundary_conditions(c: np.ndarray, c_inlet: float, c_max: float) -> np.ndarray:
    """Apply boundary conditions to concentration."""
    nx, ny = c.shape

    # Left: inlet (Dirichlet) - inject proppant
    for j in range(ny):
        c[0, j] = c_inlet

    # Right: outflow (Neumann / zero gradient)
    for j in range(ny):
        c[nx-1, j] = c[nx-2, j]

    # Top/Bottom: impermeable walls - true no-flux
    # Proppant settling is blocked by walls, so concentration can build up
    for i in range(1, nx-1):
        # Top wall: no-flux (Neumann)
        c[i, ny-1] = c[i, ny-2]
        # Bottom wall: no-flux (Neumann)
        # Proppant accumulates here due to settling
        c[i, 0] = c[i, 1]

    return c


@njit(cache=True)
def advection_step_with_settling(c: np.ndarray, u_f: np.ndarray, v_slip: np.ndarray,
                                  dx: float, dy: float, dt: float, c_max: float) -> np.ndarray:
    """
    Advection step with explicit settling term.
    ∂c/∂t + u·∂c/∂x + ∂(c·v_slip)/∂y = 0
    """
    nx, ny = c.shape
    c_new = c.copy()

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Horizontal advection (upwind for u > 0)
            if u_f[i, j] > 0:
                dc_dx = (c[i, j] - c[i-1, j]) / dx
            else:
                dc_dx = (c[i+1, j] - c[i, j]) / dx

            # Vertical settling flux (conservative form)
            # Flux at top of cell (j+1/2)
            v_top = 0.5 * (v_slip[i, j] + v_slip[i, j+1]) if j < ny-2 else v_slip[i, j]
            c_top = c[i, j] if v_top < 0 else c[i, j+1] if j < ny-2 else c[i, j]
            flux_top = c_top * v_top

            # Flux at bottom of cell (j-1/2)
            v_bot = 0.5 * (v_slip[i, j] + v_slip[i, j-1]) if j > 1 else v_slip[i, j]
            c_bot = c[i, j-1] if v_bot < 0 else c[i, j]
            flux_bot = c_bot * v_bot

            # At boundaries, no flux
            if j == 1:
                flux_bot = 0.0
            if j == ny-2:
                flux_top = 0.0

            # Update
            advection_x = u_f[i, j] * dc_dx
            settling_div = (flux_top - flux_bot) / dy

            c_new[i, j] = c[i, j] - dt * (advection_x + settling_div)

    # Clamp concentration
    for i in range(nx):
        for j in range(ny):
            if c_new[i, j] < 0:
                c_new[i, j] = 0.0
            if c_new[i, j] > c_max:
                c_new[i, j] = c_max

    return c_new


class ProppantTransportSolver:
    """Two-phase proppant transport solver with Poiseuille flow."""

    def __init__(self, sim_params: SimulationParams, phys_params: PhysicalParams):
        self.sim = sim_params
        self.phys = phys_params

        self.dx = sim_params.Lx / (sim_params.nx - 1)
        self.dy = sim_params.Ly / (sim_params.ny - 1)

        # Grid
        self.x = np.linspace(0, sim_params.Lx, sim_params.nx)
        self.y = np.linspace(0, sim_params.Ly, sim_params.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Poiseuille velocity field (fluid)
        self.u_f = create_poiseuille_velocity(
            sim_params.nx, sim_params.ny, sim_params.Ly, phys_params.U_max
        )
        self.v_f = np.zeros((sim_params.nx, sim_params.ny))

        # Pack physical params for numba
        self.phys_tuple = (
            phys_params.rho_f, phys_params.rho_p, phys_params.mu_f,
            phys_params.d_p, phys_params.g, phys_params.c_max, phys_params.U_max
        )

    def compute_proppant_velocity(self, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute proppant velocity = fluid velocity + settling."""
        # Slip velocity (settling, depends on concentration)
        v_slip = compute_slip_velocity_field(c, self.phys_tuple)

        # Proppant velocity
        u_p = self.u_f.copy()
        v_p = self.v_f + v_slip

        return u_p, v_p

    def solve(self, c0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the transport problem with continuous inlet injection.

        Args:
            c0: Initial concentration field (default: zero everywhere)

        Returns:
            times: Time array
            concentrations: Concentration fields (n_times, nx, ny)
        """
        p = self.sim
        n_steps = int(p.T / p.dt)
        n_saves = n_steps // p.save_every + 1

        times = np.zeros(n_saves)
        concentrations = np.zeros((n_saves, p.nx, p.ny))

        # Start with zero concentration (clean fracture)
        if c0 is None:
            c = np.zeros((p.nx, p.ny))
        else:
            c = c0.copy()

        # Apply inlet BC
        c = apply_boundary_conditions(c, p.c_inlet, self.phys.c_max)
        concentrations[0] = c.copy()

        save_idx = 1

        for step in range(1, n_steps + 1):
            # Compute settling velocity field
            v_slip = compute_slip_velocity_field(c, self.phys_tuple)

            # Advection + settling step
            c = advection_step_with_settling(
                c, self.u_f, v_slip,
                self.dx, self.dy, p.dt, self.phys.c_max
            )

            # Boundary conditions (continuous injection)
            c = apply_boundary_conditions(c, p.c_inlet, self.phys.c_max)

            # Save
            if step % p.save_every == 0:
                concentrations[save_idx] = c.copy()
                times[save_idx] = step * p.dt
                save_idx += 1

        return times, concentrations


def create_initial_condition(X: np.ndarray, Y: np.ndarray,
                             ic_type: str = 'empty', **kwargs) -> np.ndarray:
    """Create initial concentration field."""
    c_max = kwargs.get('c_max', 0.65)

    if ic_type == 'gaussian':
        x0 = kwargs.get('x0', 0.3)
        y0 = kwargs.get('y0', 0.5)
        sigma_x = kwargs.get('sigma_x', 0.1)
        sigma_y = kwargs.get('sigma_y', 0.15)
        amplitude = kwargs.get('amplitude', 0.35)

        c0 = amplitude * np.exp(
            -((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))
        )

    elif ic_type == 'step':
        x_thresh = kwargs.get('x_thresh', 0.3)
        amplitude = kwargs.get('amplitude', 0.3)
        c0 = np.where(X < x_thresh, amplitude, 0.0)

    elif ic_type == 'empty':
        # Empty fracture - proppant will be injected from left
        c0 = np.zeros_like(X)

    else:
        c0 = np.zeros_like(X)

    return np.clip(c0, 0, c_max * 0.95)


# Warm up JIT
def _warmup():
    sim = SimulationParams(nx=20, ny=10, T=0.05, dt=0.01, save_every=2, c_inlet=0.3)
    phys = PhysicalParams()
    solver = ProppantTransportSolver(sim, phys)
    solver.solve()

_warmup()


if __name__ == '__main__':
    import time

    # Simulate proppant injection
    sim = SimulationParams(
        nx=100, ny=50,
        T=3.0, dt=0.002,
        save_every=25,
        c_inlet=0.35
    )
    phys = PhysicalParams(
        mu_f=0.01,      # 10 mPa·s
        g=9.81,
        U_max=0.8
    )

    solver = ProppantTransportSolver(sim, phys)

    start = time.perf_counter()
    times, conc = solver.solve()  # Empty initial, inject from left
    elapsed = time.perf_counter() - start

    print(f"Solver time: {elapsed*1000:.1f} ms")
    print(f"Frames: {len(times)}")
    print(f"C range: {conc.min():.3f} - {conc.max():.3f}")
    print(f"Final max at t={times[-1]:.2f}s: {conc[-1].max():.3f}")
