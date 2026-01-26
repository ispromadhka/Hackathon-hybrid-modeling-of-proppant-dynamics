# WENO5CPU.py
import numpy as np
from typing import Tuple, Literal, Optional

class WENO5reconstructorCPU():
    """
    WENO5 (Weighted Essentially Non-Oscillatory) reconstruction scheme for CPU computations.
    
    Implements 5th order WENO scheme with different types of weight calculations
    for high-order accurate shock-capturing.
    
    Attributes
    ----------
    type : str
        Type of WENO scheme ('Z', 'M', or classical)
    Ny : int
        Number of grid points in y-direction
    Nx : int
        Number of grid points in x-direction
    dataX : np.ndarray
        Extended data array for x-direction reconstruction
    dataY : np.ndarray
        Extended data array for y-direction reconstruction
    """
    
    def __init__(
        self, 
        init_params: dict
    ) -> None:
        """
        Initialize WENO5 reconstructor with parameters.
        
        Parameters
        ----------
        init_params : dict
            Dictionary containing initialization parameters:
            - WENO_type: str - type of WENO scheme ('Z', 'M', or classical)
            - Ny: int - grid size in y-direction
            - Nx: int - grid size in x-direction
        """
        grid = init_params['grid']
        numerics = init_params['numerics']
        
        self.Ny = grid['Ny']
        self.Nx = grid['Nx']
        self.L = grid['L']
        self.H = grid['H']
        self.type = numerics.get('WENO_type', 'classical')
        
        
        self.dataX = np.zeros(shape=(self.Ny, 3 + self.Nx + 3))
        self.dataY = np.zeros(shape=(3 + self.Ny + 3, self.Nx))
        
    def precompute(
        self, 
        q: np.ndarray, 
        boundary_condition_left: np.ndarray
    ) -> None:
        """
        Precompute extended arrays with boundary conditions for reconstruction.
        
        Parameters
        ----------
        q : np.ndarray
            Input data array, shape (Ny, Nx)
        boundary_condition_left : np.ndarray  
            Boundary conditions for left boundary, shape (Ny,)
        """
        self.dataX[:, 3:-3] = q
        # Left boundary conditions
        self.dataX[:, 2] = boundary_condition_left  # j = -1
        self.dataX[:, 1] = boundary_condition_left  # j = -2
        self.dataX[:, 0] = boundary_condition_left  # j = -3
        # Right boundary conditions (zero-order extrapolation)
        self.dataX[:, -3] = q[:, -1]  # j = Nx
        self.dataX[:, -2] = q[:, -1]  # j = Nx + 1
        self.dataX[:, -1] = q[:, -1]  # j = Nx + 2
        
        self.dataY[3:-3, :] = q
        # Bottom boundary conditions (zero-order extrapolation)
        self.dataY[0, :] = q[0, :]
        self.dataY[1, :] = q[0, :]
        self.dataY[2, :] = q[0, :]
        # Top boundary conditions (zero-order extrapolation)
        self.dataY[-1, :] = q[-1, :]
        self.dataY[-2, :] = q[-1, :]
        self.dataY[-3, :] = q[-1, :]
        
    def compute_IS_K(
        self, 
        k: int, 
        projection: Literal['x', 'y']
    ) -> np.ndarray:
        """
        Compute smoothness indicator for stencil k.
        
        Parameters
        ----------
        k : int
            Stencil index (0, 1, or 2)
        projection : Literal['x', 'y']
            Axis for computation
            
        Returns
        -------
        np.ndarray
            Smoothness indicator array
        """
        
        if projection == 'x':
            if k == 0:
                # Stencil: j-2, j-1, j
                arg1 = self.dataX[:, :-4] - 2 * self.dataX[:, 1:-3] + self.dataX[:, 2:-2]
                arg2 = self.dataX[:, :-4] - 4 * self.dataX[:, 1:-3] + 3 * self.dataX[:, 2:-2]
            elif k == 1:
                # Stencil: j-1, j, j+1
                arg1 = self.dataX[:, 1:-3] - 2 * self.dataX[:, 2:-2] + self.dataX[:, 3:-1]
                arg2 = self.dataX[:, 1:-3] - self.dataX[:, 3:-1]  # Fixed missing coefficient
            elif k == 2:
                # Stencil: j, j+1, j+2
                arg1 = self.dataX[:, 2:-2] - 2 * self.dataX[:, 3:-1] + self.dataX[:, 4:]
                arg2 = 3 * self.dataX[:, 2:-2] - 4 * self.dataX[:, 3:-1] + self.dataX[:, 4:]
                
        elif projection == 'y':
            if k == 0:
                # Stencil: i-2, i-1, i
                arg1 = self.dataY[:-4, :] - 2 * self.dataY[1:-3, :] + self.dataY[2:-2, :]
                arg2 = self.dataY[:-4, :] - 4 * self.dataY[1:-3, :] + 3 * self.dataY[2:-2, :]
            elif k == 1:
                # Stencil: i-1, i, i+1
                arg1 = self.dataY[1:-3, :] - 2 * self.dataY[2:-2, :] + self.dataY[3:-1, :]
                arg2 = self.dataY[1:-3, :] - self.dataY[3:-1, :]  # Fixed missing coefficient
            elif k == 2:
                # Stencil: i, i+1, i+2
                arg1 = self.dataY[2:-2, :] - 2 * self.dataY[3:-1, :] + self.dataY[4:, :]
                arg2 = 3 * self.dataY[2:-2, :] - 4 * self.dataY[3:-1, :] + self.dataY[4:, :]
        
        # Smoothness indicator IS_k at cell j
        return 13/12 * np.square(arg1) + 1/4 * np.square(arg2)
    
    def eno_K(
        self, 
        k: int, 
        projection: Literal['x', 'y']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ENO reconstruction values for stencil k.
        
        Parameters
        ----------
        k : int
            Stencil index (0, 1, or 2)
        projection : Literal['x', 'y']
            Axis for computation
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Left/bottom and right/top reconstructed values
        """
        if projection == 'x':
            if k == 0:
                # Stencil: j-2, j-1, j
                dataL = (-1/6 * self.dataX[:, :-4] + 
                         5/6 * self.dataX[:, 1:-3] + 
                         2/6 * self.dataX[:, 2:-2])
                dataR = (2/6 * self.dataX[:, :-4] - 
                         7/6 * self.dataX[:, 1:-3] + 
                         11/6 * self.dataX[:, 2:-2])
            elif k == 1:
                # Stencil: j-1, j, j+1
                dataL = (2/6 * self.dataX[:, 1:-3] + 
                         5/6 * self.dataX[:, 2:-2] - 
                         1/6 * self.dataX[:, 3:-1])
                dataR = (-1/6 * self.dataX[:, 1:-3] + 
                         5/6 * self.dataX[:, 2:-2] + 
                         2/6 * self.dataX[:, 3:-1])
            elif k == 2:
                # Stencil: j, j+1, j+2
                dataL = (11/6 * self.dataX[:, 2:-2] - 
                         7/6 * self.dataX[:, 3:-1] + 
                         2/6 * self.dataX[:, 4:])
                dataR = (2/6 * self.dataX[:, 2:-2] + 
                         5/6 * self.dataX[:, 3:-1] - 
                         1/6 * self.dataX[:, 4:])
            return dataL, dataR
        
        elif projection == 'y':
            if k == 0:
                # Stencil: i-2, i-1, i
                dataB = (-1/6 * self.dataY[:-4, :] + 
                         5/6 * self.dataY[1:-3, :] + 
                         2/6 * self.dataY[2:-2, :])
                dataT = (2/6 * self.dataY[:-4, :] - 
                         7/6 * self.dataY[1:-3, :] + 
                         11/6 * self.dataY[2:-2, :])
            elif k == 1:
                # Stencil: i-1, i, i+1
                dataB = (2/6 * self.dataY[1:-3, :] + 
                         5/6 * self.dataY[2:-2, :] - 
                         1/6 * self.dataY[3:-1, :])
                dataT = (-1/6 * self.dataY[1:-3, :] + 
                         5/6 * self.dataY[2:-2, :] + 
                         2/6 * self.dataY[3:-1, :])
            elif k == 2:
                # Stencil: i, i+1, i+2
                dataB = (11/6 * self.dataY[2:-2, :] - 
                         7/6 * self.dataY[3:-1, :] + 
                         2/6 * self.dataY[4:, :])
                dataT = (2/6 * self.dataY[2:-2, :] + 
                         5/6 * self.dataY[3:-1, :] - 
                         1/6 * self.dataY[4:, :])
            return dataB, dataT
    
    def weights(
        self, 
        projection: Literal['x', 'y']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute nonlinear weights for WENO reconstruction.
        
        Parameters
        ----------
        projection : Literal['x', 'y']
            Axis for computation
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Left/bottom and right/top weights
        """
        # Initialize weight arrays
        if projection == 'x':
            weights_lb = np.zeros((self.Ny, 1 + self.Nx + 1, 3))
            weights_rt = np.zeros((self.Ny, 1 + self.Nx + 1, 3))
        else:  # projection == 'y'
            weights_lb = np.zeros((1 + self.Ny + 1, self.Nx, 3))
            weights_rt = np.zeros((1 + self.Ny + 1, self.Nx, 3))
        
        # Ideal weights
        dl0, dl1, dl2 = 0.3, 0.6, 0.1  # Left/bottom ideal weights
        dr0, dr1, dr2 = 0.1, 0.6, 0.3  # Right/top ideal weights
        eps = 1e-14
        
        # Compute smoothness indicators
        IS0 = self.compute_IS_K(0, projection)
        IS1 = self.compute_IS_K(1, projection)
        IS2 = self.compute_IS_K(2, projection)
        
        if self.type != 'Z':
            # Classical WENO weights
            alphal0 = dl0 / ((eps + IS0) ** 2)
            alphal1 = dl1 / ((eps + IS1) ** 2)
            alphal2 = dl2 / ((eps + IS2) ** 2)
            
            sum_l = alphal0 + alphal1 + alphal2
            weights_lb[:, :, 0] = alphal0 / sum_l
            weights_lb[:, :, 1] = alphal1 / sum_l
            weights_lb[:, :, 2] = alphal2 / sum_l
            
            alphar0 = dr0 / ((eps + IS0) ** 2)
            alphar1 = dr1 / ((eps + IS1) ** 2)
            alphar2 = dr2 / ((eps + IS2) ** 2)
            sum_r = alphar0 + alphar1 + alphar2
            
            weights_rt[:, :, 0] = alphar0 / sum_r
            weights_rt[:, :, 1] = alphar1 / sum_r
            weights_rt[:, :, 2] = alphar2 / sum_r

        if self.type == 'M':
            # Mapped WENO weights
            wl0, wl1, wl2 = weights_lb[:, :, 0], weights_lb[:, :, 1], weights_lb[:, :, 2]
            wr0, wr1, wr2 = weights_rt[:, :, 0], weights_rt[:, :, 1], weights_rt[:, :, 2]
            
            sigmal0 = wl0 * (dl0 + dl0**2 - 3*dl0*wl0 + wl0**2) / (dl0**2 + wl0*(1-2*dl0))
            sigmal1 = wl1 * (dl1 + dl1**2 - 3*dl1*wl1 + wl1**2) / (dl1**2 + wl1*(1-2*dl1))
            sigmal2 = wl2 * (dl2 + dl2**2 - 3*dl2*wl2 + wl2**2) / (dl2**2 + wl2*(1-2*dl2))
            
            sum_sigma_l = sigmal0 + sigmal1 + sigmal2
            weights_lb[:, :, 0] = sigmal0 / sum_sigma_l
            weights_lb[:, :, 1] = sigmal1 / sum_sigma_l
            weights_lb[:, :, 2] = sigmal2 / sum_sigma_l
            
            sigmar0 = wr0 * (dr0 + dr0**2 - 3*dr0*wr0 + wr0**2) / (dr0**2 + wr0*(1-2*dr0))
            sigmar1 = wr1 * (dr1 + dr1**2 - 3*dr1*wr1 + wr1**2) / (dr1**2 + wr1*(1-2*dr1))
            sigmar2 = wr2 * (dr2 + dr2**2 - 3*dr2*wr2 + wr2**2) / (dr2**2 + wr2*(1-2*dr2))
            
            sum_sigma_r = sigmar0 + sigmar1 + sigmar2
            weights_rt[:, :, 0] = sigmar0 / sum_sigma_r
            weights_rt[:, :, 1] = sigmar1 / sum_sigma_r
            weights_rt[:, :, 2] = sigmar2 / sum_sigma_r
            
        elif self.type == 'Z':
            # WENO-Z weights
            tau5 = np.abs(IS0 - IS2)
            epsZ = 1e-40
            IS0Z = (IS0 + epsZ) / (IS0 + tau5 + epsZ)
            IS1Z = (IS1 + epsZ) / (IS1 + tau5 + epsZ)
            IS2Z = (IS2 + epsZ) / (IS2 + tau5 + epsZ)
            
            sigmal0 = dl0 / IS0Z 
            sigmal1 = dl1 / IS1Z
            sigmal2 = dl2 / IS2Z
            sum_sigma_l = sigmal0 + sigmal1 + sigmal2
            weights_lb[:, :, 0] = sigmal0 / sum_sigma_l
            weights_lb[:, :, 1] = sigmal1 / sum_sigma_l
            weights_lb[:, :, 2] = sigmal2 / sum_sigma_l
            
            sigmar0 = dr0 / IS0Z 
            sigmar1 = dr1 / IS1Z
            sigmar2 = dr2 / IS2Z
            sum_sigma_r = sigmar0 + sigmar1 + sigmar2
            weights_rt[:, :, 0] = sigmar0 / sum_sigma_r
            weights_rt[:, :, 1] = sigmar1 / sum_sigma_r
            weights_rt[:, :, 2] = sigmar2 / sum_sigma_r
            
        return weights_lb, weights_rt
    
    def reconstruct(
        self, 
        projection: Literal['x', 'y']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reconstructed values at cell interfaces.
        
        Parameters
        ----------
        projection : Literal['x', 'y']
            Axis for computation
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Left/bottom and right/top reconstructed values
        """
        weights_lb, weights_rt = self.weights(projection)
        eno0L, eno0R = self.eno_K(0, projection)
        eno1L, eno1R = self.eno_K(1, projection)
        eno2L, eno2R = self.eno_K(2, projection)

        if projection == 'x':
            # Reconstructed values at left and right interfaces
            rL = (weights_lb[:, :, 0] * eno0L + 
                  weights_lb[:, :, 1] * eno1L + 
                  weights_lb[:, :, 2] * eno2L)
            rR = (weights_rt[:, :, 0] * eno0R + 
                  weights_rt[:, :, 1] * eno1R + 
                  weights_rt[:, :, 2] * eno2R)
            return rL[:, 1:], rR[:, :-1]
            
        else:  # projection == 'y'
            # Reconstructed values at bottom and top interfaces
            rB = (weights_lb[:, :, 0] * eno0L + 
                  weights_lb[:, :, 1] * eno1L + 
                  weights_lb[:, :, 2] * eno2L)
            rT = (weights_rt[:, :, 0] * eno0R + 
                  weights_rt[:, :, 1] * eno1R + 
                  weights_rt[:, :, 2] * eno2R)
            return rB[1:, :], rT[:-1, :]