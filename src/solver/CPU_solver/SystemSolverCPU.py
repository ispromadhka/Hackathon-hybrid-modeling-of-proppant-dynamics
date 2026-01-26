# SystemSolverCPU.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from .PoissonCPU import PressureSolverCPU
from .TransportCPU import TransportSolverCPU

class SolverCPU():
    """
    Main solver class for pressure and transport equations on CPU.
    
    This class combines pressure and transport solvers to simulate fluid dynamics
    with concentration-dependent viscosity and density.
    
    Attributes
    ----------
    PressureSolver : PressureSolverCPU
        Solver for pressure equations
    TransportSolver : TransportSolverCPU
        Solver for transport equations
    Q : np.ndarray
        Concentration field over time, shape (Nt, Ny, Nx)
    P : np.ndarray
        Pressure field over time, shape (Nt, Ny, Nx)
    Vx : np.ndarray
        Velocity in x-direction over time, shape (Nt, Ny, Nx+1)
    Vy : np.ndarray
        Velocity in y-direction over time, shape (Nt, Ny+1, Nx)
    w : np.ndarray
        Width field over time, shape (Nt, Ny+1, Nx)
    """
    
    def __init__(
        self, 
        init_params: dict
    ) -> None:
        """
        Initialize solver with parameters from dictionary.
        
        Parameters
        ----------
        init_params : dict
            Dictionary containing all initialization parameters
        """
        # Initialize parameters from dictionary
        grid = init_params['grid']
        physics = init_params['physics']
        numerics = init_params['numerics']
        bc = init_params['boundary_conditions']
        
        self.Ny = grid['Ny']
        self.Nx = grid['Nx']
        self.L = grid['L']
        self.H = grid['H']
        self.Tmax = grid['Tmax']
        self.dT = grid['dT']
        self.Nt = int(self.Tmax/self.dT) + 1
        
        self.beta = physics['beta']
        self.mu0 = physics['mu0']
        self.cmax = physics['cmax']
        self.rho1 = physics['rho1']
        self.rho2 = physics['rho2']
        self.g = physics['g']
        self.r = physics['r']
        
        self.CFL = numerics['CFL']
        self.rk_stages = numerics['rk_stages']
        self.prefer_CG = numerics['prefer_CG']
        self.prefer_async_CG = numerics['prefer_async_CG']
        self.async_check_interval = numerics['async_check_interval']
        
        self.q_in = bc['q_in']
        self.q_out = bc['q_out']
        self.c_in_times = np.asarray(bc['c_in_times'])  # Convert to numpy array
        self.c_in_arr = np.asarray(bc['c_in_arr'])      # Convert to numpy array
        
        self.dx = self.L / self.Nx
        self.dy = self.H / self.Ny
        self.Vst = -2*self.r**2/9*(self.rho2-self.rho1)*self.g/self.mu0

        # Initialize solvers
        self.PressureSolver = PressureSolverCPU(init_params)
        self.TransportSolver = TransportSolverCPU(init_params)
        
        # Initialize arrays
        self.Q = np.zeros(shape=(self.Nt, self.Ny, self.Nx))
        self.P = np.zeros(shape=(self.Nt, self.Ny, self.Nx))
        self.w = physics['w']*np.ones(shape=(self.Ny, self.Nx))
        
        self.Vx = np.zeros(shape=(self.Nt, self.Ny, self.Nx+1))
        self.Vy = np.zeros(shape=(self.Nt, self.Ny+1, self.Nx))
        
        # Initialize time tracking
        self.step = 0
        self.times = np.zeros(self.Nt)  

        self.Q[0] = physics['c0'] * self.w
        self.P[0], self.Vx[0], self.Vy[0] = self.pressure_and_velocities(
            self.Q[0], np.zeros_like(self.Q[0]).reshape(-1)
        )
    
    def mu(
        self, 
        c: np.ndarray
    ) -> np.ndarray:
        """
        Compute concentration-dependent viscosity.
        
        Parameters
        ----------
        c : np.ndarray
            Concentration field, 2D array, shape (Ny, Nx)
            
        Returns
        -------
        np.ndarray
            Viscosity field, same shape as input
        """
        return self.mu0 * (1 - np.minimum(c/ self.cmax, 0.99) ) ** (-self.beta)

    def rho(
        self, 
        c: np.ndarray
    ) -> np.ndarray:
        """
        Compute concentration-dependent density.
        
        Parameters
        ----------
        c : np.ndarray
            Concentration field, 2D array, shape (Ny, Nx)
            
        Returns
        -------
        np.ndarray
            Density field, same shape as input
        """
        return self.rho1 + (self.rho2 - self.rho1) * c

    def pressure_rhs(
        self, 
        KrhoG_center: np.ndarray,
        wcVs_center: np.ndarray
    ) -> np.ndarray:
        """
        Compute right-hand side for pressure equation.
        
        Parameters
        ----------
        KrhoG_center : np.ndarray
            Centered K*rho*g term
        wcVs_center : np.ndarray
            Centered w*c*Vs term
            
        Returns
        -------
        np.ndarray
            Right-hand side vector for pressure equation
        """
        div_KrhoG = (KrhoG_center[1:] - KrhoG_center[:-1]) / self.dy
        div_wcVs = (wcVs_center[1:] - wcVs_center[:-1]) / self.dy
        f = - (div_KrhoG + div_wcVs)
        rhs = np.zeros(shape=(self.Ny, self.Nx))
        rhs[1:-1] = f * self.dx * self.dy 
        rhs[:, 0] += self.q_in * self.dy
        rhs[:, -1] += self.q_out  * self.dy
        rhs[0] += -KrhoG_center[0] * self.dx - wcVs_center[0] * self.dx # - KrhoG_1/2/dy * dx*dy
        rhs[-1] += KrhoG_center[-1] * self.dx + wcVs_center[-1] * self.dx
#         print(-KrhoG_center[0] * self.dx - wcVs_center[0] * self.dx)
        
        return rhs
    
    def Harmonic(self, x):
        return 2 * x[1:] * x[:-1] / (x[1:] + x[:-1])
    
    def Mean(self, x):
        return (x[1:] + x[:-1]) / 2.0
    
    def pressure_and_velocities(
        self, 
        c: np.ndarray,
        p0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pressure and velocity fields for given concentration.
        
        Parameters
        ----------
        c : np.ndarray
            Concentration field, 2D array, shape (Ny, Nx)
        p0 : np.ndarray
            Initial approximation for CG solver, 2D array, shape (Ny, Nx)
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Pressure field, x-velocity, y-velocity
        """
        Lambda = -self.w**2 / (12 * self.mu(c))
        LambdaX = self.Harmonic(Lambda.T).T  # ij -> Lambda i+1/2, j
        LambdaY = self.Harmonic(Lambda)  # ij -> Lambda i, j+1/2
        
        LambdaRhoG_Y = self.Harmonic(Lambda*self.rho(c))*self.g
        
        K = Lambda * self.w
        KrhoG_Y = self.Harmonic(K*self.rho(c))*self.g 
        h_pl = (1 - np.minimum(c, self.cmax * 0.99) / self.cmax) ** 5
        Vs_Y = self.Vst*self.Harmonic(h_pl)           
        wcVs_Y = self.Mean(c)*self.Vst*self.Harmonic(self.w*h_pl)
        rhs_matrix = self.pressure_rhs(KrhoG_Y, wcVs_Y)  
        self.PressureSolver.update_matrix(K)
        pressure = self.PressureSolver.solve(
            RHS_matrix=rhs_matrix, 
            sol0=p0,
            use_async=self.prefer_async_CG, 
            use_CG=self.prefer_CG, 
            check_interval=self.async_check_interval
        )
        
        vx = np.zeros((self.Ny, self.Nx + 1))
        vx[:, 1:-1] = LambdaX * (pressure[:, 1:] - pressure[:, :-1]) / self.dx
        vx[:, 0] = self.q_in/self.w[:, 0]
        vx[:, -1] = -self.q_out/self.w[:, -1]
        
        vy = np.zeros(shape=(self.Ny+1, self.Nx))
        vy[1:-1] += LambdaY * (pressure[1:] - pressure[:-1]) / self.dy + Vs_Y + LambdaRhoG_Y
        vy[0] = 0
        vy[-1] = 0
        
        return pressure - np.min(pressure), vx, vy

    def make_step(
        self, 
        Q: np.ndarray, 
        P: np.ndarray, 
        Vx: np.ndarray, 
        Vy: np.ndarray, 
        current_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Make one time step using appropriate Runge-Kutta method.
        
        Parameters
        ----------
        Q : np.ndarray
            Current concentration field
        P : np.ndarray
            Current pressure field
        Vx : np.ndarray
            Current x-velocity field
        Vy : np.ndarray
            Current y-velocity field
        current_time : float
            Current simulation time
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
            Updated fields and new time
        """
        tau = self.CFL * min((self.dx / np.max(np.abs(Vx))),
                             (self.dy / np.max(np.abs(Vy))))
        dt = min(tau, self.dT * (self.step + 1) - current_time)
#         print(dt)
        
        c_in_index_now = np.searchsorted(self.c_in_times, current_time, side='right')
        c_in_value = self.c_in_arr[c_in_index_now]    
        
        new_time = current_time + dt
        
        if self.rk_stages == 1:
            newQ, newP, newVx, newVy = self.rk1_cycle(Q, P, Vx, Vy, c_in_value, dt)
        if self.rk_stages == 2:
            newQ, newP, newVx, newVy = self.rk2_cycle(Q, P, Vx, Vy, c_in_value, dt)
        if self.rk_stages == 3:
            newQ, newP, newVx, newVy = self.rk3_cycle(Q, P, Vx, Vy, c_in_value, dt)
        return newQ, newP, newVx, newVy, new_time

    def solve(
        self, 
        tmax: float = 1.0
    ) -> None:
        """
        Solve the system up to specified maximum time.
        
        Parameters
        ----------
        tmax : float, optional
            Maximum simulation time, default is 1.0
        """
        q, p, vx, vy, current_time = (
            self.Q[self.step], self.P[self.step],
            self.Vx[self.step], self.Vy[self.step],
            self.times[self.step]
        )
        local_step = 0
        total_steps = min(int((tmax - current_time) / self.dT), self.Nt-1)
        progress_bar = tqdm(total=total_steps, desc="Solving", unit="step")

        while current_time < tmax:
            q, p, vx, vy, current_time = self.make_step(q, p, vx, vy, current_time)
            local_step += 1

            if np.abs(current_time - self.dT * (self.step + 1)) <= 1e-5:  # Use np.abs
                self.step += 1
                self.times[self.step] = current_time
                self.Q[self.step], self.P[self.step], self.Vx[self.step], self.Vy[self.step] = q, p, vx, vy
                progress_bar.update(1)

                if self.step == self.Nt - 1:
                    print("Nt-limit")
                    break

        progress_bar.close()
        print("done")

    def rk1_cycle(
        self, 
        Q: np.ndarray, 
        P: np.ndarray, 
        Vx: np.ndarray, 
        Vy: np.ndarray, 
        c_in_value: float,
        dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        First-order Runge-Kutta step.
        
        Parameters
        ----------
        Q : np.ndarray
            Current concentration field
        P : np.ndarray
            Current pressure field
        Vx : np.ndarray
            Current x-velocity field
        Vy : np.ndarray
            Current y-velocity field
        current_time : float
            Current simulation time
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Updated fields
        """
        q_left = np.where(self.q_in == 0, Q[:,0], c_in_value*self.w[:,0])    
        
        q_RK1 = self.TransportSolver.RK(Q, Q, dt, Vx, Vy, q_left, 1) 
        if self.beta == 0 and (self.g==0 or self.rho1==self.rho2):
            p_RK1, vx_RK1, vy_RK1 = self.P[0], self.Vx[0], self.Vy[0]
        else:
            p_RK1, vx_RK1, vy_RK1 = self.pressure_and_velocities(q_RK1 / self.w, P)
            
        return q_RK1, p_RK1, vx_RK1, vy_RK1
    
    def rk2_cycle(
        self, 
        Q: np.ndarray, 
        P: np.ndarray, 
        Vx: np.ndarray,
        Vy: np.ndarray, 
        c_in_value: float,
        dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Second-order Runge-Kutta step.
        
        Parameters
        ----------
        Q : np.ndarray
            Current concentration field
        P : np.ndarray
            Current pressure field
        Vx : np.ndarray
            Current x-velocity field
        Vy : np.ndarray
            Current y-velocity field
        current_time : float
            Current simulation time
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Updated fields
        """
        q_left = np.where(self.q_in == 0, Q[:,0], c_in_value*self.w[:,0])    
        q_RK1 = self.TransportSolver.RK(Q, Q, dt, Vx, Vy, q_left, 1) 
        if self.beta == 0 and (self.g==0 or self.rho1==self.rho2):
            q_left = np.where(self.q_in == 0, q_RK1[:,0], c_in_value*self.w[:,0])    
            q_RK2 = self.TransportSolver.RK(Q, q_RK1, dt, self.Vx[0], self.Vy[0], q_left, 2)            
            p_RK2, vx_RK2, vy_RK2 = self.P[0], self.Vx[0], self.Vy[0]
        else:
            p_RK1, vx_RK1, vy_RK1 = self.pressure_and_velocities(q_RK1 / self.w, P)
            
            q_left = np.where(self.q_in == 0, q_RK1[:,0], c_in_value*self.w[:,0])    
            q_RK2 = self.TransportSolver.RK(Q, q_RK1, dt, vx_RK1, vy_RK1, q_left, 2)
            p_RK2, vx_RK2, vy_RK2 = self.pressure_and_velocities(q_RK2 / self.w, p_RK1)

        return q_RK2, p_RK2, vx_RK2, vy_RK2
    
    def rk3_cycle(
        self, 
        Q: np.ndarray, 
        P: np.ndarray, 
        Vx: np.ndarray,
        Vy: np.ndarray, 
        c_in_value: float,
        dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Third-order Runge-Kutta step.
        
        Parameters
        ----------
        Q : np.ndarray
            Current concentration field
        P : np.ndarray
            Current pressure field
        Vx : np.ndarray
            Current x-velocity field
        Vy : np.ndarray
            Current y-velocity field
        current_time : float
            Current simulation time
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Updated fields
        """
        q_left = np.where(self.q_in == 0, Q[:,0], c_in_value*self.w[:,0])    
        
        q_RK1 = self.TransportSolver.RK(Q, Q, dt, Vx, Vy, q_left, 1) 
        if self.beta == 0 and (self.g==0 or self.rho1==self.rho2):
            q_left = np.where(self.q_in == 0, q_RK1[:,0], c_in_value*self.w[:,0])    
            q_RK2 = self.TransportSolver.RK(Q, q_RK1, dt, self.Vx[0], self.Vy[0], q_left, 2)      
            
            q_left = np.where(self.q_in == 0, q_RK2[:,0], c_in_value*self.w[:,0])    
            q_RK3 = self.TransportSolver.RK(Q, q_RK2, dt, self.Vx[0], self.Vy[0], q_left, 3)   
            p_RK3, vx_RK3, vy_RK3 = self.P[0], self.Vx[0], self.Vy[0]
        else:
            p_RK1, vx_RK1, vy_RK1 = self.pressure_and_velocities(q_RK1 / self.w, P)
            
            q_left = np.where(self.q_in == 0, q_RK1[:,0], c_in_value*self.w[:,0])    
            q_RK2 = self.TransportSolver.RK(Q, q_RK1, dt, vx_RK1, vy_RK1, q_left, 2)
            p_RK2, vx_RK2, vy_RK2 = self.pressure_and_velocities(q_RK2 / self.w, p_RK1)
            
            q_left = np.where(self.q_in == 0, q_RK2[:,0], c_in_value*self.w[:,0])    
            q_RK3 = self.TransportSolver.RK(Q, q_RK2, dt, vx_RK2, vy_RK2, q_left, 3)
            p_RK3, vx_RK3, vy_RK3 = self.pressure_and_velocities(q_RK3 / self.w, p_RK2)

        return q_RK3, p_RK3, vx_RK3, vy_RK3
    
    def data_to_numpy(
        self
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return arrays for analysis/plotting (already numpy arrays).
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Q, P, Vx, Vy as numpy arrays
        """
        return (
            self.Q[:self.step], 
            self.P[:self.step], 
            self.Vx[:self.step], 
            self.Vy[:self.step]
        )