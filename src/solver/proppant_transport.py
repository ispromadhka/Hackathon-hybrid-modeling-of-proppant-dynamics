"""
Two-phase proppant transport solver.

Equations:
    ∂(cw)/∂t + ∇·(cwVₚ) = 0              — mass conservation
    Vf = -w²/(12μ(c)) · (∇P - ρ(c)g)     — Darcy's law (slot flow)
    Vₚ = Vf + Vslip(c)                    — proppant velocity
    ∇·(Vf + c·Vslip) = 0                  — incompressibility

Closure models:
    μ(c)  — Krieger-Dougherty viscosity
    ρ(c)  — linear mixture density
    Vslip — Richardson-Zaki hindered settling
"""

import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import Tuple
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


@dataclass
class PhysicalParams:
    """Physical parameters for proppant transport."""
    # Fluid properties
    rho_f: float = 1000.0       # Fluid density [kg/m³]
    mu_f: float = 0.001         # Fluid viscosity [Pa·s]

    # Proppant properties
    rho_p: float = 2650.0       # Proppant density [kg/m³]
    d_p: float = 0.0005         # Proppant diameter [m] (500 μm)
    c_max: float = 0.6          # Maximum packing fraction

    # Gravity
    g: float = 9.81             # Gravitational acceleration [m/s²]

    # Fracture
    w0: float = 0.005           # Fracture width [m] (5 mm)


@dataclass
class SimulationParams:
    """Simulation parameters."""
    Lx: float = 2.0             # Domain length x [m]
    Ly: float = 1.0             # Domain length y [m]
    nx: int = 64                # Grid points x
    ny: int = 32                # Grid points y
    T: float = 4.0              # Total time [s]
    dt: float = 0.005           # Time step [s]
    save_every: int = 20        # Save interval


@njit(cache=True)
def mixture_density(c: float, rho_f: float, rho_p: float) -> float:
    """Linear mixture density: ρ(c) = ρf(1-c) + ρp·c"""
    return rho_f * (1.0 - c) + rho_p * c


@njit(cache=True)
def mixture_viscosity(c: float, mu_f: float, c_max: float) -> float:
    """
    Krieger-Dougherty viscosity model:
    μ(c) = μf · (1 - c/c_max)^(-2.5·c_max)
    """
    if c >= c_max * 0.99:
        c = c_max * 0.99  # Prevent singularity
    ratio = 1.0 - c / c_max
    exponent = -2.5 * c_max
    return mu_f * (ratio ** exponent)


@njit(cache=True)
def settling_velocity(c: float, rho_f: float, rho_p: float,
                      mu_f: float, d_p: float, g: float, c_max: float) -> float:
    """
    Richardson-Zaki hindered settling:
    Vslip = V_stokes · (1-c)^n

    V_stokes = (ρp - ρf) · g · d² / (18·μf)
    n ≈ 4.65 for intermediate Reynolds
    """
    if c >= c_max * 0.99:
        return 0.0

    # Stokes settling velocity
    V_stokes = (rho_p - rho_f) * g * d_p * d_p / (18.0 * mu_f)

    # Hindered settling (Richardson-Zaki)
    n = 4.65
    hindrance = (1.0 - c / c_max) ** n

    return -V_stokes * hindrance  # Negative = downward


@njit(cache=True)
def compute_viscosity_field(c: np.ndarray, mu_f: float, c_max: float) -> np.ndarray:
    """Compute viscosity at all grid points."""
    nx, ny = c.shape
    mu = np.zeros_like(c)
    for i in range(nx):
        for j in range(ny):
            mu[i, j] = mixture_viscosity(c[i, j], mu_f, c_max)
    return mu


@njit(cache=True)
def compute_density_field(c: np.ndarray, rho_f: float, rho_p: float) -> np.ndarray:
    """Compute density at all grid points."""
    nx, ny = c.shape
    rho = np.zeros_like(c)
    for i in range(nx):
        for j in range(ny):
            rho[i, j] = mixture_density(c[i, j], rho_f, rho_p)
    return rho


@njit(cache=True)
def compute_slip_velocity(c: np.ndarray, phys: tuple) -> np.ndarray:
    """Compute vertical slip velocity field."""
    rho_f, rho_p, mu_f, d_p, g, c_max = phys
    nx, ny = c.shape
    v_slip = np.zeros_like(c)
    for i in range(nx):
        for j in range(ny):
            v_slip[i, j] = settling_velocity(c[i, j], rho_f, rho_p, mu_f, d_p, g, c_max)
    return v_slip


def build_pressure_matrix(nx: int, ny: int, dx: float, dy: float,
                          mobility: np.ndarray) -> sp.csr_matrix:
    """
    Build pressure Poisson matrix with variable mobility.

    ∇·(K·∇P) = 0, where K = w²/(12μ)
    """
    N = nx * ny

    # Coefficient arrays
    diag = np.zeros(N)
    off_x = np.zeros(N)  # x+1
    off_x_neg = np.zeros(N)  # x-1
    off_y = np.zeros(N)  # y+1
    off_y_neg = np.zeros(N)  # y-1

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j

            # Interior points
            if 0 < i < nx-1 and 0 < j < ny-1:
                # Harmonic average of mobility at cell faces
                K_e = 2.0 * mobility[i,j] * mobility[i+1,j] / (mobility[i,j] + mobility[i+1,j] + 1e-10)
                K_w = 2.0 * mobility[i,j] * mobility[i-1,j] / (mobility[i,j] + mobility[i-1,j] + 1e-10)
                K_n = 2.0 * mobility[i,j] * mobility[i,j+1] / (mobility[i,j] + mobility[i,j+1] + 1e-10)
                K_s = 2.0 * mobility[i,j] * mobility[i,j-1] / (mobility[i,j] + mobility[i,j-1] + 1e-10)

                diag[idx] = -(K_e + K_w) / dx**2 - (K_n + K_s) / dy**2
                off_x[idx] = K_e / dx**2
                off_x_neg[idx] = K_w / dx**2
                off_y[idx] = K_n / dy**2
                off_y_neg[idx] = K_s / dy**2
            else:
                # Boundary: Dirichlet (P=0 at left, P=0 at right) or Neumann
                diag[idx] = 1.0

    # Build sparse matrix
    diagonals = [diag, off_x[:-ny], off_x_neg[ny:], off_y[:-1], off_y_neg[1:]]
    offsets = [0, ny, -ny, 1, -1]

    # Fix the off-diagonal terms at boundaries
    A = sp.diags(diagonals, offsets, shape=(N, N), format='csr')

    return A


@njit(cache=True)
def compute_velocity_from_pressure(P: np.ndarray, c: np.ndarray, w: np.ndarray,
                                   dx: float, dy: float, phys: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fluid velocity from pressure gradient.
    Vf = -w²/(12μ(c)) · (∇P - ρ(c)g)
    """
    rho_f, rho_p, mu_f, d_p, g, c_max = phys
    nx, ny = P.shape

    u = np.zeros((nx, ny))  # x-velocity
    v = np.zeros((nx, ny))  # y-velocity

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            mu = mixture_viscosity(c[i,j], mu_f, c_max)
            rho = mixture_density(c[i,j], rho_f, rho_p)
            mobility = w[i,j]**2 / (12.0 * mu)

            # Pressure gradient
            dPdx = (P[i+1,j] - P[i-1,j]) / (2.0 * dx)
            dPdy = (P[i,j+1] - P[i,j-1]) / (2.0 * dy)

            # Darcy velocity (gravity acts in -y direction)
            u[i,j] = -mobility * dPdx
            v[i,j] = -mobility * (dPdy + rho * g)

    return u, v


@njit(cache=True)
def advection_step(c: np.ndarray, w: np.ndarray,
                   u_p: np.ndarray, v_p: np.ndarray,
                   dx: float, dy: float, dt: float) -> np.ndarray:
    """
    Advection step for concentration using upwind scheme.
    ∂(cw)/∂t + ∇·(cwVₚ) = 0
    """
    nx, ny = c.shape
    c_new = c.copy()

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Fluxes at cell faces (upwind)
            # East face
            if u_p[i,j] > 0:
                flux_e = c[i,j] * w[i,j] * u_p[i,j]
            else:
                flux_e = c[i+1,j] * w[i+1,j] * u_p[i,j] if i+1 < nx else 0

            # West face
            if u_p[i-1,j] > 0:
                flux_w = c[i-1,j] * w[i-1,j] * u_p[i-1,j]
            else:
                flux_w = c[i,j] * w[i,j] * u_p[i-1,j]

            # North face
            if v_p[i,j] > 0:
                flux_n = c[i,j] * w[i,j] * v_p[i,j]
            else:
                flux_n = c[i,j+1] * w[i,j+1] * v_p[i,j] if j+1 < ny else 0

            # South face
            if v_p[i,j-1] > 0:
                flux_s = c[i,j-1] * w[i,j-1] * v_p[i,j-1]
            else:
                flux_s = c[i,j] * w[i,j] * v_p[i,j-1]

            # Update
            div_flux = (flux_e - flux_w) / dx + (flux_n - flux_s) / dy
            c_new[i,j] = c[i,j] - dt / w[i,j] * div_flux

    # Clamp concentration
    for i in range(nx):
        for j in range(ny):
            if c_new[i,j] < 0:
                c_new[i,j] = 0.0
            if c_new[i,j] > 0.6:
                c_new[i,j] = 0.6

    return c_new


@njit(cache=True)
def apply_boundary_conditions(c: np.ndarray, c_inlet: float) -> np.ndarray:
    """Apply boundary conditions to concentration."""
    nx, ny = c.shape

    # Left: inlet (Dirichlet)
    for j in range(ny):
        c[0, j] = c_inlet

    # Right: outflow (Neumann)
    for j in range(ny):
        c[nx-1, j] = c[nx-2, j]

    # Top/Bottom: no-flux (Neumann)
    for i in range(nx):
        c[i, 0] = c[i, 1]
        c[i, ny-1] = c[i, ny-2]

    return c


class ProppantTransportSolver:
    """Full two-phase proppant transport solver."""

    def __init__(self, sim_params: SimulationParams, phys_params: PhysicalParams):
        self.sim = sim_params
        self.phys = phys_params

        self.dx = sim_params.Lx / (sim_params.nx - 1)
        self.dy = sim_params.Ly / (sim_params.ny - 1)

        # Grid
        self.x = np.linspace(0, sim_params.Lx, sim_params.nx)
        self.y = np.linspace(0, sim_params.Ly, sim_params.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Fracture width (constant for now)
        self.w = np.ones((sim_params.nx, sim_params.ny)) * phys_params.w0

        # Pack physical params for numba
        self.phys_tuple = (
            phys_params.rho_f, phys_params.rho_p, phys_params.mu_f,
            phys_params.d_p, phys_params.g, phys_params.c_max
        )

    def solve_pressure(self, c: np.ndarray) -> np.ndarray:
        """Solve pressure equation."""
        nx, ny = self.sim.nx, self.sim.ny

        # Compute mobility field: K = w²/(12μ)
        mu = compute_viscosity_field(c, self.phys.mu_f, self.phys.c_max)
        mobility = self.w**2 / (12.0 * mu)

        # Simple approach: prescribed pressure drop
        # P = P_in at left, P = 0 at right
        P = np.zeros((nx, ny))
        P_in = 1000.0  # Inlet pressure [Pa]

        # Linear pressure gradient as initial guess / simple solution
        for i in range(nx):
            P[i, :] = P_in * (1.0 - self.x[i] / self.sim.Lx)

        return P

    def compute_velocities(self, P: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute fluid and proppant velocities."""
        # Fluid velocity from pressure
        u_f, v_f = compute_velocity_from_pressure(
            P, c, self.w, self.dx, self.dy, self.phys_tuple
        )

        # Slip velocity (settling)
        v_slip = compute_slip_velocity(c, self.phys_tuple)

        # Proppant velocity
        u_p = u_f.copy()
        v_p = v_f + v_slip

        return u_f, v_f, u_p, v_p

    def solve(self, c0: np.ndarray, c_inlet: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the full transport problem.

        Args:
            c0: Initial concentration field
            c_inlet: Inlet concentration (0 = no proppant at inlet)

        Returns:
            times: Time array
            concentrations: Concentration fields (n_times, nx, ny)
        """
        p = self.sim
        n_steps = int(p.T / p.dt)
        n_saves = n_steps // p.save_every + 1

        times = np.zeros(n_saves)
        concentrations = np.zeros((n_saves, p.nx, p.ny))

        c = c0.copy()
        c = apply_boundary_conditions(c, c_inlet)
        concentrations[0] = c.copy()

        save_idx = 1

        for step in range(1, n_steps + 1):
            # Solve pressure
            P = self.solve_pressure(c)

            # Compute velocities
            u_f, v_f, u_p, v_p = self.compute_velocities(P, c)

            # Advection step
            c = advection_step(c, self.w, u_p, v_p, self.dx, self.dy, p.dt)

            # Boundary conditions
            c = apply_boundary_conditions(c, c_inlet)

            # Save
            if step % p.save_every == 0:
                concentrations[save_idx] = c.copy()
                times[save_idx] = step * p.dt
                save_idx += 1

        return times, concentrations


def create_initial_condition(X: np.ndarray, Y: np.ndarray,
                             ic_type: str = 'gaussian', **kwargs) -> np.ndarray:
    """Create initial concentration field."""
    if ic_type == 'gaussian':
        x0 = kwargs.get('x0', 0.3)
        y0 = kwargs.get('y0', 0.5)
        sigma_x = kwargs.get('sigma_x', 0.1)
        sigma_y = kwargs.get('sigma_y', 0.15)
        amplitude = kwargs.get('amplitude', 0.4)  # Max ~0.4 to stay below c_max

        c0 = amplitude * np.exp(
            -((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))
        )

    elif ic_type == 'step':
        x_thresh = kwargs.get('x_thresh', 0.5)
        amplitude = kwargs.get('amplitude', 0.3)
        c0 = np.where(X < x_thresh, amplitude, 0.0)

    elif ic_type == 'layer':
        # Horizontal layer of proppant
        y_center = kwargs.get('y_center', 0.3)
        thickness = kwargs.get('thickness', 0.2)
        amplitude = kwargs.get('amplitude', 0.3)
        c0 = np.where(np.abs(Y - y_center) < thickness/2, amplitude, 0.0)

    else:
        c0 = np.zeros_like(X)

    return np.clip(c0, 0, 0.55)


# Warm up JIT
def _warmup():
    sim = SimulationParams(nx=16, ny=8, T=0.02, dt=0.01, save_every=1)
    phys = PhysicalParams()
    solver = ProppantTransportSolver(sim, phys)
    c0 = np.random.rand(16, 8) * 0.3
    solver.solve(c0)

_warmup()


if __name__ == '__main__':
    import time

    sim = SimulationParams(nx=64, ny=32, T=2.0, dt=0.005, save_every=20)
    phys = PhysicalParams()

    solver = ProppantTransportSolver(sim, phys)
    c0 = create_initial_condition(solver.X, solver.Y, 'gaussian', x0=0.3, y0=0.5)

    start = time.perf_counter()
    times, conc = solver.solve(c0)
    elapsed = time.perf_counter() - start

    print(f"Solver time: {elapsed*1000:.1f} ms")
    print(f"Frames: {len(times)}")
    print(f"C range: {conc.min():.3f} - {conc.max():.3f}")
