# TransportCPU.py
import numpy as np
from typing import Tuple, Literal

from .WENO5CPU import WENO5reconstructorCPU
from .TVD_CPU import TVDreconstructorCPU

class TransportSolverCPU():
    """
    Transport equation solver for CPU computations.
    
    Solves transport equations using WENO5 or TVD reconstruction schemes
    with Runge-Kutta time integration.
    
    Attributes
    ----------
    WENOreconstructor : WENO5reconstructorCPU
        WENO5 reconstruction scheme
    TVDreconstructor : TVDreconstructorCPU
        TVD reconstruction scheme
    Ny : int
        Number of grid points in y-direction
    Nx : int
        Number of grid points in x-direction
    L : float
        Domain length in x-direction
    H : float
        Domain height in y-direction
    rk_stages : int
        Number of Runge-Kutta stages
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    use_WENO : bool
        Flag to use WENO reconstruction (True) or TVD (False)
    """
    
    def __init__(
        self, 
        init_params: dict
    ) -> None:
        """
        Initialize transport solver with parameters.
        
        Parameters
        ----------
        init_params : dict
            Dictionary containing initialization parameters with keys:
            - grid: dict - grid parameters (Nt, Ny, Nx, L, H)
            - numerics: dict - numerical parameters (rk_stages, use_WENO, etc.)
        """
        # Initialize reconstruction schemes
        self.WENOreconstructor = WENO5reconstructorCPU(init_params)
        self.TVDreconstructor = TVDreconstructorCPU(init_params)
        
        # Extract parameters from structured dictionary
        grid = init_params['grid']
        numerics = init_params['numerics']
        
        # Grid parameters
        self.Ny = grid['Ny']
        self.Nx = grid['Nx']
        self.L = grid['L']
        self.H = grid['H']
        
        # Numerical parameters
        self.rk_stages = numerics['rk_stages']
        self.use_WENO = numerics.get('use_WENO', True)  # Default to WENO
        
        # Grid spacing
        self.dx = self.L / self.Nx
        self.dy = self.H / self.Ny
        
    def boundary_vals(
        self,
        q: np.ndarray,
        boundary_conditions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reconstructed values at cell boundaries.
        
        Parameters
        ----------
        q : np.ndarray
            Concentration field, shape (Ny, Nx)
        boundary_conditions : np.ndarray
            Boundary conditions for left boundary, shape (Ny,)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Left, right, bottom, and top reconstructed values
        """
        if self.use_WENO:
            # Use WENO5 reconstruction
            self.WENOreconstructor.precompute(q, boundary_conditions)
            Q_left, Q_right = self.WENOreconstructor.reconstruct('x')
            Q_bottom, Q_top = self.WENOreconstructor.reconstruct('y')
        else:
            # Use TVD reconstruction
            self.TVDreconstructor.precompute(q, boundary_conditions)
            Q_left = self.TVDreconstructor.left()
            Q_right = self.TVDreconstructor.right()
            Q_bottom = self.TVDreconstructor.bottom()
            Q_top = self.TVDreconstructor.top()
            
        return Q_left, Q_right, Q_bottom, Q_top
    
    def Lu(
        self,
        Q: np.ndarray,
        Vx: np.ndarray,
        Vy: np.ndarray,
        boundary_conditions: np.ndarray
    ) -> np.ndarray:
        """
        Compute the spatial discretization operator L(u).
        
        Parameters
        ----------
        q : np.ndarray
            Concentration field, shape (Ny, Nx)
        Vx : np.ndarray
            Velocity in x-direction, shape (Ny, Nx+1)
        Vy : np.ndarray
            Velocity in y-direction, shape (Ny+1, Nx)
        dt : float
            Time step
        boundary_conditions : np.ndarray
            Boundary conditions for left boundary, shape (Ny,)
            
        Returns
        -------
        np.ndarray
            Spatial operator L(u), shape (Ny, Nx)
        """
        # Get reconstructed values at cell boundaries
        Q_left, Q_right, Q_bottom, Q_top = self.boundary_vals(Q, boundary_conditions)
        


        # Fluxes in x-direction
        # fxp: flux from right face to current cell
        fxp = (-Q_left[:, 1:] * np.minimum(Vx[:, 1:], 0) + 
                Q_right[:, :-1] * np.maximum(Vx[:, :-1], 0))
        
        # fxm: flux from left face to current cell  
        fxm = (-Q_left[:, :-1] * np.minimum(Vx[:, :-1], 0) + 
                Q_right[:, 1:] * np.maximum(Vx[:, 1:], 0))
        
        # Fluxes in y-direction
        # fyp: flux from top face to current cell
        fyp = (-Q_bottom[1:] * np.minimum(Vy[1:], 0) + 
                Q_top[:-1] * np.maximum(Vy[:-1], 0))
        
        # fym: flux from bottom face to current cell
        fym = (-Q_bottom[:-1] * np.minimum(Vy[:-1], 0) + 
                Q_top[1:] * np.maximum(Vy[1:], 0))
        
        # Compute spatial operator
        Lu = (fxp - fxm) / self.dx + (fyp - fym) / self.dy
        
        return Lu
    
    def RK(
        self,
        Q: np.ndarray,
        Q_RK: np.ndarray,
        dt: float,
        Vx: np.ndarray,
        Vy: np.ndarray,
        boundary_conditions: np.ndarray,
        RKstep: Literal[1, 2, 3]
    ) -> np.ndarray:
        """
        Perform one step of Runge-Kutta time integration.
        
        First step RK:
            q^n(1) = q^n + dt*L(q^n)
        
        Second step RK:
            q^n(2) = 3*q^n/4 + (q^n(1) + dt*L(q^n(1)))/4
            
        Third step RK:
            q^n+1 = q^n/3 + (q^n(2) + dt*L(q^n(2)))*2/3
            
        Parameters
        ----------
        Q : np.ndarray
            Initial solution q^n, shape (Ny, Nx)
        Q_RK : np.ndarray  
            Current RK stage solution, shape (Ny, Nx)
        dt : float
            Time step size
        Vx : np.ndarray
            Velocity in x-direction, shape (Ny, Nx+1)
        Vy : np.ndarray
            Velocity in y-direction, shape (Ny+1, Nx)
        boundary_conditions: np.ndarray
            Concentration at left boundary
        RKstep : Literal[1, 2, 3]
            Current Runge-Kutta step (1, 2, or 3)
            
        Returns
        -------
        np.ndarray
            Updated solution at next RK stage, shape (Ny, Nx)
        """
        
        # Compute spatial operator
        Lu = self.Lu(Q_RK, Vx, Vy, boundary_conditions)
        
        # Perform Runge-Kutta step
        if self.rk_stages == 1:
            if RKstep == 1:
                new_Q = Q * 0 + (Q_RK + dt * Lu)
                
        elif self.rk_stages == 2:
            if RKstep == 1:
                new_Q = Q * 0 + (Q_RK + dt * Lu)
            elif RKstep == 2:
                new_Q = 1/2 * Q + 1/2 * (Q_RK + dt * Lu)
                
        elif self.rk_stages == 3:
            if RKstep == 1:
                new_Q = Q * 0 + (Q_RK + dt * Lu)
            elif RKstep == 2:
                new_Q = 3/4 * Q + 1/4 * (Q_RK + dt * Lu)
            elif RKstep == 3:
                new_Q = 1/3 * Q + 2/3 * (Q_RK + dt * Lu)
        
        # Ensure non-negative concentration
        return np.maximum(new_Q, 0)