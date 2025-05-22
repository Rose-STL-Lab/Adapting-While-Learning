import numpy as np
from tqdm import tqdm
import json


def convert_dict_values_to_float(input_dict):
    return {k: float(v) for k, v in input_dict.items()}


class DiffusionSolver:
    def __init__(self):
        self.Nx = 10
        self.Ny = 10
        self.Nt = 1000
        self.initial_condition = None
        self.boundary_condition = None
        self.solution = None
        self.D = None
        self.L = None
        self.Ly = None
        self.T = None
        self.dimension = None

    def setup_and_solve(
        self,
        dimension,
        D,
        T,
        L=None,
        Lx=None,
        Ly=None,
        ic_type=None,
        ic_params=None,
        bc_type=None,
        bc_params=None,
        solution_type=None,
        solution_x=None,
        solution_y=None,
    ):
        # Set dimension
        if dimension not in ["1D", "2D"]:
            raise ValueError("Dimension must be either '1D' or '2D'.")
        self.dimension = dimension

        # Set DLT parameters
        self.D = float(D)
        self.T = float(T)
        if self.dimension == "1D":
            if L is None:
                raise ValueError("L must be provided for 1D problems.")
            self.L = float(L)
        elif self.dimension == "2D":
            if Lx is None or Ly is None:
                raise ValueError("Both Lx and Ly must be provided for 2D problems.")
            self.Lx = float(Lx)
            self.Ly = float(Ly)

        # Set initial condition
        self.set_initial_condition(ic_type, ic_params)

        self.set_boundary_condition(bc_type, bc_params)

        ret_str = "Simulation Result:\n"

        if self.dimension == "1D":
            ret_str += str(self.solve_1d_diffusion())
        else:
            ret_str += str(self.solve_2d_diffusion())

        ret_str += "\nThe result for solution: "

        # Return the requested solution type
        if solution_type == "get_max_value":
            ret_str += str(self.get_max_value())
        elif solution_type == "get_min_value":
            ret_str += str(self.get_min_value())
        elif solution_type == "get_mean_value":
            ret_str += str(self.get_mean_value())
        elif solution_type == "get_value_at_point":
            if solution_x is None:
                raise ValueError("solution_x must be provided for get_value_at_point")
            if self.dimension == "2D" and solution_y is None:
                raise ValueError(
                    "solution_y must be provided for 2D get_value_at_point"
                )
            ret_str += str(self.get_value_at_point(solution_x, solution_y))
        elif solution_type == "get_max_gradient":
            ret_str += str(self.get_max_gradient())
        elif solution_type == "get_equilibrium_time":
            ret_str += str(self.get_equilibrium_time())
        else:
            raise ValueError(f"Unsupported solution type: {solution_type}")

        return ret_str

    def check_stability(self):
        if self.dimension == "1D":
            dx = self.L / (self.Nx - 1)
            dt = self.T / self.Nt
            alpha = self.D * dt / dx**2
            if alpha > 0.5:
                self.Nt = int(0.5 * self.D * self.T / dx**2) + 1
                print(f"Adjusted Nt to {self.Nt} for stability")
        elif self.dimension == "2D":
            dx = self.Lx / (self.Nx - 1)
            dy = self.Ly / (self.Ny - 1)
            dt = self.T / self.Nt
            alpha_x = self.D * dt / dx**2
            alpha_y = self.D * dt / dy**2
            if alpha_x + alpha_y > 0.5:
                self.Nt = int(0.5 * self.D * self.T * (1 / dx**2 + 1 / dy**2)) + 1
                print(f"Adjusted Nt to {self.Nt} for stability")

    def set_initial_condition(self, ic_type, ic_params):
        ic_type = ic_type.lower()
        if not isinstance(ic_params, dict):
            if "```" in ic_params:
                ic_params = ic_params.split("```json")[1].split("```")[0]
            params = json.loads(ic_params)
        else:
            params = ic_params

        params = convert_dict_values_to_float(params)

        if ic_type == "constant":
            value = params["value"]
            if self.dimension == "1D":
                # 对于 1D，初始条件只依赖于 x
                self.initial_condition = lambda x: np.full_like(x, value)
            else:
                # 对于 2D，初始条件依赖于 x 和 y
                self.initial_condition = lambda x, y: np.full((len(y), len(x)), value)

        elif ic_type == "step":
            left_value = params["left_value"]
            right_value = params["right_value"]
            step_position = float(self.L / 2)  # 转换为浮点数，确保兼容
            if self.dimension == "1D":
                self.initial_condition = lambda x: np.where(
                    x < step_position, left_value, right_value
                )
            else:
                self.initial_condition = lambda x, y: np.where(
                    x < step_position, left_value, right_value
                )

        elif ic_type == "checkerboard":
            value1 = params["value1"]
            value2 = params["value2"]
            cell_size_x = params["cell_size_x"]
            cell_size_y = params["cell_size_y"]
            if self.dimension == "1D":
                raise ValueError(
                    "Checkerboard initial condition is not applicable for 1D problems."
                )
            else:
                self.initial_condition = lambda x, y: np.where(
                    (x // cell_size_x + y // cell_size_y) % 2, value1, value2
                )
        if self.initial_condition:
            return "Successfully set initial condition"
        else:
            return "Initial condition not set"

    def set_boundary_condition(self, bc_type, bc_params):
        bc_type = bc_type.lower()
        self.check_stability()
        if not isinstance(bc_params, dict):
            if "```" in bc_params:
                bc_params = bc_params.split("```json")[1].split("```")[0]
            params = json.loads(bc_params)
        else:
            params = bc_params

        params = convert_dict_values_to_float(params)

        # Handle 1D boundary conditions
        if self.dimension == "1D":
            if bc_type == "dirichlet":
                left_val = params.get("left_value", 0)  # Default to 0 if not provided
                right_val = params.get("right_value", 0)  # Default to 0 if not provided
                self.boundary_condition = lambda u: np.concatenate(
                    ([left_val], u[1:-1], [right_val])
                )
            elif bc_type == "neumann":
                self.boundary_condition = lambda u: np.concatenate(
                    ([u[1]], u[1:-1], [u[-2]])
                )
            elif bc_type == "periodic":
                self.boundary_condition = lambda u: np.concatenate(
                    ([u[-2]], u[1:-1], [u[1]])
                )
            return "Successfully set 1D boundary condition"

        # Handle 2D boundary conditions
        elif self.dimension == "2D":
            if bc_type == "dirichlet":
                left_val = params.get("left_value", 0)
                right_val = params.get("right_value", 0)
                bottom_val = params.get("bottom_value", 0)
                top_val = params.get("top_value", 0)
                self.boundary_condition = lambda u: np.pad(
                    u[1:-1, 1:-1],
                    ((1, 1), (1, 1)),
                    mode="constant",
                    constant_values=((bottom_val, top_val), (left_val, right_val)),
                )
            elif bc_type == "neumann":
                self.boundary_condition = lambda u: np.pad(
                    u[1:-1, 1:-1], ((1, 1), (1, 1)), mode="edge"
                )
            elif bc_type == "periodic":
                self.boundary_condition = lambda u: np.pad(
                    u[1:-1, 1:-1], ((1, 1), (1, 1)), mode="wrap"
                )

        if self.boundary_condition:
            return "Successfully set 2D boundary condition"
        else:
            return "Boundary condition not set"

    def solve_1d_diffusion(self):
        self.check_stability()
        self.dimension = "1D"
        D, L, T = self.D, self.L, self.T
        dx = L / (self.Nx - 1)
        dt = T / self.Nt
        alpha = D * dt / dx**2

        x = np.linspace(0, L, self.Nx)
        u = self.initial_condition(x)

        for _ in range(1, self.Nt):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + alpha * (u[2:] - 2 * u[1:-1] + u[:-2])
            u_new = self.boundary_condition(u_new)
            u = u_new

        self.solution = u
        return u

    def solve_2d_diffusion(
        self,
    ):
        self.check_stability()
        self.dimension = "2D"
        D, Lx, Ly, T = self.D, self.Lx, self.Ly, self.T
        dx = Lx / (self.Nx - 1)
        dy = Ly / (self.Ny - 1)
        dt = T / self.Nt
        alpha_x = D * dt / dx**2
        alpha_y = D * dt / dy**2

        x = np.linspace(0, Lx, self.Nx)
        y = np.linspace(0, Ly, self.Ny)
        X, Y = np.meshgrid(x, y)
        u = self.initial_condition(X, Y)

        for _ in range(1, self.Nt):
            u_new = u.copy()
            u_new[1:-1, 1:-1] = (
                u[1:-1, 1:-1]
                + alpha_x * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
                + alpha_y * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
            )
            u_new = self.boundary_condition(u_new)
            u = u_new

        self.solution = u
        return u

    def get_max_value(self):
        return np.max(self.solution)

    def get_min_value(self):
        return np.min(self.solution)

    def get_mean_value(self):
        return np.mean(self.solution)

    def get_value_at_point(self, x, y=None):
        if x:
            x = int(x)
        if y:
            y = int(y)
        if self.solution is None:
            raise ValueError("Solution has not been computed yet.")

        print(self.solution)

        if self.dimension == "1D":
            # Handle 1D case
            x_values = np.linspace(0, self.L, self.Nx)
            return str(np.interp(x, x_values, self.solution))

        elif self.dimension == "2D":
            if y is None:
                raise ValueError("For 2D, both x and y coordinates must be provided.")

            x_index = min(int(x / self.Lx * self.Nx), self.Nx - 1)
            y_index = min(int(y / self.Ly * self.Ny), self.Ny - 1)

            # Ensure the solution is 2D
            if self.solution.ndim != 2:
                raise ValueError(
                    f"Expected 2D solution, but got {self.solution.ndim}D array."
                )

            return self.solution[y_index, x_index]
        else:
            raise ValueError(f"Unknown dimension: {self.dimension}")

    def get_max_gradient(self):
        if self.dimension == "1D":
            gradient = np.gradient(self.solution, self.L / self.Nx)
        else:
            gradient = np.gradient(self.solution, self.Lx / self.Nx, self.Ly / self.Ny)
        return round(np.max(np.abs(gradient)), 2)

    def get_equilibrium_time(self, threshold=0.01):
        times = np.linspace(0, self.T, 10)
        for t in times:
            if self.dimension == "1D":
                u = self.solve_1d_diffusion(self.D, self.L, t)
            else:
                u = self.solve_2d_diffusion(self.D, self.Lx, self.Ly, t)
            if np.max(u) - np.min(u) < threshold * np.mean(u):
                return t
        return self.T