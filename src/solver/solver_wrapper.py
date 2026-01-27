"""
Wrapper for CPU_solver for visualization and training.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from CPU_solver.SystemSolverCPU import SolverCPU


class ProppantSolver:
    """
    High-level wrapper for proppant transport solver.

    Uses the full two-phase model:
    - Pressure equation with concentration-dependent viscosity
    - Transport with TVD/WENO schemes and RK time integration
    - Settling velocity with hindered settling
    """

    def __init__(
        self,
        # Grid
        nx: int = 60,
        ny: int = 30,
        Lx: float = 60.0,
        Ly: float = 30.0,
        T: float = 100.0,
        dT: float = 5.0,
        # Physics
        c_inlet: float = 0.35,
        mu0: float = 0.001,
        beta: float = 2.5,
        cmax: float = 0.635,
        rho_fluid: float = 1000.0,
        rho_proppant: float = 2650.0,
        g: float = 9.81,
        r_particle: float = 0.0002,
        w0: float = 0.01,
        # Flow
        Q_inlet: float = 0.05,
        inlet_fraction: float = 0.5,  # Half of height (reasonable inlet)
        inlet_position: float = 0.5,  # Center position (0=bottom, 1=top)
        # Injection pattern
        injection_duration: float = None,  # None = continuous injection
        c_in_times: np.ndarray = None,
        c_in_arr: np.ndarray = None,
        # Numerics
        cfl: float = 0.8,
        rk_stages: int = 2,
        use_weno: bool = False,
        lim_type: str = 'koren',
    ):
        """
        Initialize proppant transport solver.

        Parameters
        ----------
        nx, ny : int
            Grid resolution
        Lx, Ly : float
            Domain size [m]
        T : float
            Total simulation time [s]
        dT : float
            Output time step [s]
        c_inlet : float
            Inlet proppant concentration (0-0.6)
        mu0 : float
            Base fluid viscosity [Pa·s]
        beta : float
            Viscosity exponent (Krieger-Dougherty: mu = mu0*(1-c/cmax)^(-beta))
        cmax : float
            Maximum packing fraction
        rho_fluid, rho_proppant : float
            Densities [kg/m³]
        g : float
            Gravity [m/s²]
        r_particle : float
            Particle radius [m]
        w0 : float
            Fracture width [m]
        Q_inlet : float
            Total inlet flow rate [m²/s]
        inlet_fraction : float
            Fraction of height for inlet (default 0.1 = 10%)
        injection_duration : float
            Duration of proppant injection [s] (default 0.5s)
        c_in_times : np.ndarray, optional
            Times at which inlet concentration changes (overrides injection_duration)
        c_in_arr : np.ndarray, optional
            Concentration values for each time interval
        cfl : float
            CFL number
        rk_stages : int
            Runge-Kutta stages (1, 2, or 3)
        use_weno : bool
            Use WENO5 (True) or TVD (False)
        lim_type : str
            TVD limiter type: 'koren', 'superbee', 'minmod', 'vanleer'
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.dT = dT
        self.w0 = w0

        # Coordinate arrays
        self.x = np.linspace(0, Lx, nx, endpoint=False) + Lx / nx / 2
        self.y = np.linspace(0, Ly, ny, endpoint=False) + Ly / ny / 2

        # Inlet profile: slit at left boundary (x=0)
        # inlet_fraction = fraction of height that is inlet (0.5 = half height)
        # inlet_position = center position of inlet (0.5 = middle of domain)
        chi = Ly * inlet_fraction  # inlet height
        y = self.y
        inlet_center = Ly * inlet_position

        # Create inlet velocity profile (uniform across inlet slit)
        q_in = Q_inlet / chi * np.where(np.abs(y - inlet_center) < chi/2, 1.0, 0.0)

        # Outlet: uniform across entire right boundary (mass conservation)
        q_out = -Q_inlet / Ly

        # Configuration
        self.config = {
            'grid': {
                'Tmax': T,
                'Ny': ny,
                'Nx': nx,
                'L': Lx,
                'H': Ly,
                'dT': dT,
            },
            'physics': {
                'c0': np.zeros((ny, nx)),
                'w': w0 * np.ones((ny, nx)),
                'beta': beta,
                'mu0': mu0,
                'cmax': cmax,
                'rho1': rho_fluid,
                'rho2': rho_proppant,
                'g': g,
                'r': r_particle,
            },
            'numerics': {
                'CFL': cfl,
                'rk_stages': rk_stages,
                'lim_type': lim_type,
                'kappa': -1,
                'WENO_type': 'Z',
                'use_WENO': use_weno,
                'prefer_CG': False,
                'prefer_async_CG': False,
                'async_check_interval': 50,
                'eps': 1e-10,
                'max_iter': 10000,
            },
            'boundary_conditions': {
                'q_in': q_in,
                'q_out': q_out,
                # Injection pattern:
                # - If c_in_times/c_in_arr provided: use them
                # - If injection_duration is None: continuous injection (entire simulation)
                # - Otherwise: pulse injection for injection_duration seconds
                'c_in_times': c_in_times if c_in_times is not None else (
                    np.array([T * 100]) if injection_duration is None  # Continuous
                    else np.array([injection_duration])  # Pulse
                ),
                'c_in_arr': c_in_arr if c_in_arr is not None else np.array([c_inlet, 0.0]),
            }
        }

        # Store parameters for data generation
        self.params = {
            'c_inlet': c_inlet,
            'mu0': mu0,
            'g': g,
            'r_particle': r_particle,
            'Q_inlet': Q_inlet,
            'inlet_fraction': inlet_fraction,
            'inlet_position': inlet_position,
            'injection_duration': injection_duration,
            'rk_stages': rk_stages,
            'lim_type': lim_type,
        }

        # Initialize solver
        self._solver = SolverCPU(self.config)

    def solve(self, tmax: float = None) -> tuple:
        """
        Run simulation.

        Returns
        -------
        times : np.ndarray
            Time points
        concentrations : np.ndarray
            Concentration field (n_times, ny, nx)
        """
        if tmax is None:
            tmax = self.T

        self._solver.solve(tmax=tmax)

        # Get results
        n_steps = self._solver.step + 1
        times = self._solver.times[:n_steps]

        # Convert Q (mass) to concentration c = Q/w
        w = self.config['physics']['w']
        cmax = self.config['physics']['cmax']
        concentrations = self._solver.Q[:n_steps] / w

        # Clip to physical range [0, cmax]
        concentrations = np.clip(concentrations, 0, cmax)

        return times, concentrations

    def get_velocities(self) -> tuple:
        """Get velocity fields (Vx, Vy)."""
        n_steps = self._solver.step + 1
        return self._solver.Vx[:n_steps], self._solver.Vy[:n_steps]

    def get_pressure(self) -> np.ndarray:
        """Get pressure field."""
        n_steps = self._solver.step + 1
        return self._solver.P[:n_steps]


def run_test():
    """Test the solver."""
    import time

    print("Testing ProppantSolver...")

    t0 = time.time()
    solver = ProppantSolver(
        nx=50, ny=25,
        Lx=60, Ly=30,
        T=50.0, dT=5.0,
        c_inlet=0.35,
        g=9.81,
        r_particle=0.0003,
        mu0=0.001,
        inlet_fraction=0.5,    # Half height inlet
        inlet_position=0.5,    # Centered
        injection_duration=None,  # Continuous injection
    )
    print(f"Init: {time.time()-t0:.2f}s")

    t0 = time.time()
    times, conc = solver.solve()
    print(f"Solve: {time.time()-t0:.2f}s")

    print(f"Times: {len(times)} steps")
    print(f"Concentration: {conc.shape}")
    print(f"C range: [{conc.min():.4f}, {conc.max():.4f}]")

    return times, conc


if __name__ == '__main__':
    run_test()
