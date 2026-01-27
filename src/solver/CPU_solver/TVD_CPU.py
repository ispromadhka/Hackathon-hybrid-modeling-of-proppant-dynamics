# TVD_CPU.py
import numpy as np


class TVDreconstructorCPU():
    """
    TVD (Total Variation Diminishing) reconstructor for CPU computations.
    
    Implements various slope limiters for high-resolution shock-capturing schemes.
    
    Attributes
    ----------
    Ny : int
        Number of grid points in y-direction
    Nx : int
        Number of grid points in x-direction
    lim_type : str
        Type of slope limiter to use
    data : np.ndarray
        Primary data array
    datadXl, datadXr, datadYt, datadYb : np.ndarray
        Slope arrays in different directions
    kappa : float
        Parameter for reconstruction scheme
    kappaM, kappaP : float
        Derived parameters from kappa
    """
    
    def __init__(
        self, 
        init_params: dict
    ) -> None:
        """
        Initialize TVD reconstructor with parameters.
        
        Parameters
        ----------
        init_params : dict
            Dictionary containing initialization parameters:
            - Ny: int - grid size in y-direction
            - Nx: int - grid size in x-direction  
            - lim_type: str - type of limiter
            - kappa: float - reconstruction parameter
        """
        grid = init_params['grid']
        numerics = init_params['numerics']
        
        self.Ny = grid['Ny']
        self.Nx = grid['Nx']
        self.L = grid['L']
        self.H = grid['H']
        self.kappa = numerics["kappa"]
        self.lim_type = numerics["lim_type"]
        
        self.dataX = np.zeros(shape=(self.Ny, 2 + self.Nx + 2))
        self.dataY = np.zeros(shape=(2 + self.Ny + 2, self.Nx))
        self.Lim1tildaX = np.zeros(shape=(self.Ny, 1 + self.Nx + 1))
        self.Lim2tildaX = np.zeros(shape=(self.Ny, 1 + self.Nx + 1))
        self.Lim1tildaY = np.zeros(shape=(1 + self.Ny + 1, self.Nx))
        self.Lim2tildaY = np.zeros(shape=(1 + self.Ny + 1, self.Nx))
        self.kappaM = 0.25 * (1.0 - self.kappa)
        self.kappaP = 0.25 * (1.0 + self.kappa)
        
    def precompute(
        self, 
        q: np.ndarray, 
        boundary_condition_left: np.ndarray
    ) -> None:
        """
        Precompute slopes for reconstruction.
        
        Parameters
        ----------
        q : np.ndarray
            Input data array, shape (Ny, Nx)
        boundary_conditions_left : np.ndarray  
            Boundary conditions for left boundary, shape (Ny,)
        """
        self.dataX[:,2:-2] = q # j = 0, Nx-1
        self.dataX[:,1] = boundary_condition_left # j=-1
        self.dataX[:,0] = boundary_condition_left # j=-2
        self.dataX[:,-2] = q[:,-1] # j = Nx
        self.dataX[:,-1] = q[:,-1] # j=Nx+1 
        
        self.dataY[2:-2] = q
        self.dataY[0] = q[0]
        self.dataY[1] = q[0]
        self.dataY[-1] = q[-1]
        self.dataY[-2] = q[-1]
        
        # Compute slopes in x-direction
        datadX = self.dataX[:, 1:] - self.dataX[:, :-1]  # dq_j = q_j+1 - q_j, j= -2, ..., Nx, len = Nx+3
        #                           q_j+1 - q_j,  q_j - q_j-1
        self.Lim1tildaX = self.lim(datadX[:,1:], datadX[:,:-1]) # lim(du_i, du_i-1), i=-1, Nx
        #                          q_j - q_j-1,  q_j+1 - q_j
        self.Lim2tildaX = self.lim(datadX[:,:-1], datadX[:,1:]) # lim(du_i-1, du_i), i=-1, Nx
        
        # Compute slopes in y-direction  
        datadY = self.dataY[1:] - self.dataY[:-1]  # dq_i = q_i+1 - q_i, i= -2, Ny, len = Ny+3
        #                           q_j+1 - q_j,  q_j - q_j-1
        self.Lim1tildaY = self.lim(datadY[1:], datadY[:-1]) # lim(du_i, du_i-1), i=-1, Nx
        #                          q_j - q_j-1,  q_j+1 - q_j
        self.Lim2tildaY = self.lim(datadY[:-1], datadY[1:]) # lim(du_i-1, du_i), i=-1, Nx

    def left(self) -> np.ndarray:
        """
        Get left reconstructed values.
         
        Returns
        -------
        np.ndarray
            Left reconstructed values for cells 0, ..., Nx, lenght=Nx+1 
        """
        dataL = self.dataX[:,1:-1] - self.kappaM * self.Lim1tildaX - self.kappaP * self.Lim2tildaX
        return dataL[:, 1:]
    
    def right(self) -> np.ndarray:
        """
        Get right reconstructed values.
        
        Returns
        -------
        np.ndarray
            Right reconstructed values for cells -1, ..., Nx-1, lenght=Nx+1
        """
        dataR = self.dataX[:,1:-1] + self.kappaM * self.Lim2tildaX + self.kappaP * self.Lim1tildaX
        return dataR[:, :-1]
    
    def bottom(self) -> np.ndarray:
        """
        Get bottom reconstructed values.
        
        Returns
        -------
        np.ndarray
            Bottom reconstructed values for cells 0, ..., Ny, lenght=Ny+1 
        """
        dataB = self.dataY[1:-1] - self.kappaM * self.Lim1tildaY - self.kappaP * self.Lim2tildaY

        return dataB[1:]
    
    def top(self) -> np.ndarray:
        """
        Get top reconstructed values.
        
        Returns
        -------
        np.ndarray
            Top reconstructed values for cells -1, ..., Ny-1, lenght=Ny+1
        """
        dataT = self.dataY[1:-1] + self.kappaM * self.Lim2tildaY + self.kappaP * self.Lim1tildaY
        return dataT[:-1]
        
    def lim(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply slope limiter function directly using x and y.
        
        Parameters
        ----------
        x : np.ndarray
            First slope array, u_j - u_{j-1}
        y : np.ndarray  
            Second slope array, u_{j+1} - u_j
            
        Returns
        -------
        np.ndarray
            Limited slopes
        """
        eps = 1e-8
        
        if self.lim_type == 'minmod':
            # minmod: 0.5*(sign(x)+sign(y)) * min(|x|,|y|)
            return 0.5 * (np.sign(x) + np.sign(y)) * np.minimum(np.abs(x), np.abs(y))
        
        elif self.lim_type == 'albada1':
            # van Albada 1: x*(x*y + y²)/(x² + y²)
            return x * (x*y + y**2) / (x**2 + y**2 + eps)
        
        elif self.lim_type == 'albada2':
            # van Albada 2: 2*x²*y/(x² + y²)
            return 2 * x**2 * y / (x**2 + y**2 + eps)
        
        elif self.lim_type == 'koren':
            # Koren: max(0, min(2y, (x+2y)/3, 2x)) для x>0
            result = np.zeros_like(x)
            
            # Для положительных x
            mask_pos = x > 0
            if np.any(mask_pos):
                x_pos = x[mask_pos]
                y_pos = y[mask_pos]
                term = np.minimum(2*y_pos, (x_pos + 2*y_pos)/3)
                term = np.minimum(term, 2*x_pos)
                result[mask_pos] = np.maximum(0, term)
            
            # Для отрицательных x
            mask_neg = x < 0
            if np.any(mask_neg):
                x_neg = x[mask_neg]
                y_neg = y[mask_neg]
                term = np.maximum(2*y_neg, (x_neg + 2*y_neg)/3)
                term = np.maximum(term, 2*x_neg)
                result[mask_neg] = np.minimum(0, term)
            
            # Для x = 0 остается 0
            return result
        
        elif self.lim_type == 'superbee':
            # Superbee: max(0, min(2y, x), min(y, 2x)) для x>0
            result = np.zeros_like(x)
            
            # Для положительных x
            mask_pos = x > 0
            if np.any(mask_pos):
                x_pos = x[mask_pos]
                y_pos = y[mask_pos]
                term1 = np.minimum(2*y_pos, x_pos)
                term2 = np.minimum(y_pos, 2*x_pos)
                result[mask_pos] = np.maximum(0, np.maximum(term1, term2))
            
            # Для отрицательных x
            mask_neg = x < 0
            if np.any(mask_neg):
                x_neg = x[mask_neg]
                y_neg = y[mask_neg]
                term1 = np.maximum(2*y_neg, x_neg)
                term2 = np.maximum(y_neg, 2*x_neg)
                result[mask_neg] = np.minimum(0, np.minimum(term1, term2))
            
            return result
        
        elif self.lim_type == 'leer':
            # van Leer: (y*|x| + |y|*x) / (|x| + |y|)
            abs_x = np.abs(x)
            abs_y = np.abs(y)
            return (y*abs_x + abs_y*x) / (abs_x + abs_y + eps)
        
        elif self.lim_type == "zero":
            # first order
            return 0
        
        else:
            return x