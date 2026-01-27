# PoissonCPU.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import warnings

class PressureSolverCPU:
    def __init__(self, init_params):
        grid = init_params['grid']
        numerics = init_params['numerics']

        self.Ny = grid['Ny']
        self.Nx = grid['Nx']
        self.L = grid['L']
        self.H = grid['H']
        self.dx = self.L/self.Nx
        self.dy = self.H/self.Ny
        self.tau = self.dy/self.dx
        self.eps = numerics["eps"]
        self.max_iter = numerics["max_iter"]
        self.matrix_size = self.Nx*self.Ny

        self.matrix, self.rows, self.cols = self.init_matrix()
        self._iterations = 0
        self._rel_residual = 0.0
        self._abs_residual = 0

    def init_matrix(self) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Инициализация матрицы"""
        # Создаем индексную сетку
        index_grid = np.arange(self.Ny * self.Nx, dtype=np.int32).reshape(self.Ny, self.Nx)

        # Верхние и нижние соседи
        row_top = index_grid[:-1, :].ravel()
        row_bottom = index_grid[1:, :].ravel()

        # Левые и правые соседи
        row_left = index_grid[:, :-1].ravel()
        row_right = index_grid[:, 1:].ravel()

        # Центральные элементы
        row_center = index_grid.ravel()

        data = np.concatenate([
            np.ones_like(row_top),
            np.ones_like(row_bottom),
            np.ones_like(row_left),
            np.ones_like(row_right),
            np.ones_like(row_center)
        ], dtype=np.float64)

        # Индексы должны точно соответствовать GPU версии
        row = np.concatenate([row_top, row_bottom, row_left, row_right, row_center], dtype="int64")
        col = np.concatenate([
            row_top + self.Nx,     # верхние соседи
            row_bottom - self.Nx,  # нижние соседи
            row_left + 1,          # правые соседи
            row_right - 1,         # левые соседи
            row_center             # диагональ
        ], dtype="int64")

        return sp.csr_matrix((data, (row, col)), shape=(self.matrix_size, self.matrix_size)), row, col

    def compute_data(self, Lambda: np.ndarray) -> np.ndarray:
        """Вычисление данных матрицы"""
        LambdaY = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        LambdaX = np.zeros((self.Ny, self.Nx), dtype=np.float64)

        # Вычисляем гармонические средние
        LambdaY[:-1, :] = 2 / (1 / Lambda[:-1, :] + 1 / Lambda[1:, :])
        LambdaX[:, :-1] = 2 / (1 / Lambda[:, :-1] + 1 / Lambda[:, 1:])


        # Верхние соседи
        data_top = (LambdaY[:-1, :] / self.tau).ravel()

        # Нижние соседи
        data_bottom = (LambdaY[:-1, :] / self.tau).ravel()

        # Левые соседи
        data_left = (LambdaX[:, :-1] * self.tau).ravel()

        # Правые соседи
        data_right = (LambdaX[:, :-1] * self.tau).ravel()

        # Центральные элементы
        m_center = np.zeros(self.Ny * self.Nx, dtype=np.float64)

        # Вертикальные связи (Y-направление)
        m_center[:-self.Nx] -= (LambdaY[:-1, :] / self.tau).ravel()  # верхние
        m_center[self.Nx:] -= (LambdaY[:-1, :] / self.tau).ravel()   # нижние

        # Горизонтальные связи (X-направление)
        LambdaX_full_tau = (LambdaX * self.tau).ravel()
        m_center[:-1] -= LambdaX_full_tau[:-1]  # левые
        m_center[1:] -= LambdaX_full_tau[:-1]   # правые

        # Конкатенация в том же порядке
        data = np.concatenate([data_top, data_bottom, data_left, data_right, m_center])
        return data

    def update_matrix(self, Lambda: np.ndarray) -> None:
        """Обновляет матрицу системы"""
        # Получаем отсортированные данные
        new_data = self.compute_data(Lambda)

        # Обновляем данные матрицы
        self.matrix = sp.csr_matrix((new_data, (self.rows, self.cols)), shape=(self.matrix_size, self.matrix_size))

    def solve(self,
                RHS_matrix: np.ndarray,
                sol0: np.ndarray,
                use_CG: bool = False,
                use_async: bool = False,
                check_interval: int = 50) -> np.ndarray:
        """
        Решает систему для давления

        Parameters
        ----------
        rhs_matrix: np.ndarray
            Правая часть
        p0: np.ndarray, optional
            Начальное приближение
        use_sparse: bool
            Игнорируется в CPU версии
        check_interval: int
            Интервал проверки сходимости
        """
        if use_CG:
            return self._solve_CG_basic(RHS_matrix, sol0, check_interval)


        return self._solve_sparse(RHS_matrix, sol0, check_interval)

    def _solve_CG_basic(self,
                       rhs_matrix: np.ndarray,
                       p0: np.ndarray,
                       check_interval: int = 10) -> np.ndarray:
        """
        Базовая версия метода сопряженных градиентов
        """
        rhs_vec = rhs_matrix.reshape(-1)
        rhs_norm = np.sqrt(np.dot(rhs_vec, rhs_vec))
        x_j = p0.reshape(-1) * 1.0

        # Инициализация: r_j = rhs_vec - self.matrix @ x_j
        r_j = rhs_vec - self.matrix @ x_j

        # p_j = r_j
        p_j = r_j.copy()

        r_dot_r = np.dot(r_j, r_j)

        i = 0
        for i in range(1, self.max_iter + 1):
            # Умножение матрицы на вектор
            A_p_j = self.matrix @ p_j

            # Вычисление шага
            alpha = r_dot_r / np.dot(p_j, A_p_j)

            # Обновление решения и невязки
            x_j += alpha * p_j
            r_j -= alpha * A_p_j

            # Новая норма
            r_dot_r_new = np.dot(r_j, r_j)

            # Проверяем сходимость только каждые check_interval итераций
            if i % check_interval == 0 or i == self.max_iter:
                r_norm = np.sqrt(r_dot_r_new)

                if r_norm < self.eps*rhs_norm:
                    break

            # Обновление направления
            beta = r_dot_r_new / r_dot_r
            p_j = r_j + beta * p_j
            r_dot_r = r_dot_r_new

        # Сохраняем статистику
        self._iterations = i
        self._abs_residual = float(r_norm)
        self._rel_residual = self._abs_residual/rhs_norm

        if self._abs_residual >= self.eps:
            print("WARNING: CG didnt converge")

        return x_j.reshape(self.Ny, self.Nx)


    def _solve_sparse(self,
                    rhs_matrix: np.ndarray,
                    p0: np.ndarray,
                    check_interval: int = 10) -> np.ndarray:
        """Computes pressure field for given mobility and sources matrix using scipy  solver\n
        Parameters
        ----------
        Lambda: np.ndarray
            mobility, 2D array, shape = (Ny, Nx)
        rhs_matrix: np.ndarray
            matrix of sources and boundary conditions, 2D array, shape = (Ny, Nx)
            """
        rhs_vec = rhs_matrix.reshape(-1)
        rhs_norm = np.sqrt(np.dot(rhs_vec, rhs_vec))
        warnings.filterwarnings("error")
        try:
            solution_vec = spsolve(self.matrix, rhs_vec)
            self._iterations = 0
            r = self.matrix@solution_vec-rhs_vec
            self._abs_residual = np.dot(r,r)
            self._rel_residual = self._abs_residual/rhs_norm
            solution_matrix = solution_vec.reshape(self.Ny, self.Nx)
        except:
            print('spsolve error ')
            print('emergency cg')
            solution_matrix = self._solve_CG_basic(rhs_matrix, p0, check_interval)
        return solution_matrix

    def get_solver_stats(self) -> dict:
        """
        Возвращает статистику последнего решения
        """
        return {
            'iterations': self._iterations,
            'rel_residual': self._rel_residual,
            "abs_residual" :self._abs_residual,
            'converged': self._rel_residual < self.eps
        }

    def get_matrix(self) -> np.ndarray:
        """
        Возвращает матрицу в плотном формате
        """
        return self.matrix.todense()

    def set_tolerance(self, eps: float):
        """Устанавливает точность решения"""
        self.eps = eps

    def set_max_iterations(self, max_iter: int):
        """Устанавливает максимальное число итераций"""
        self.max_iter = max_iter
