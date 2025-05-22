import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof

data_path = "./input_data/"

min_co2 = 0.0
max_co2 = 9500


def normalize_co2(data):
    return data / max_co2


def un_normalize_co2(data):
    return data * max_co2


min_ch4 = 0.0
max_ch4 = 0.8


def normalize_ch4(data):
    return data / max_ch4


def un_normalize_ch4(data):
    return data * max_ch4


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def modify_grid(grid, line_points, modification_method, value):
    if len(line_points) == 1:
        x, y = line_points[0]
        if modification_method == "solute":
            grid[x][y] = value
        elif modification_method == "add":
            grid[x][y] += value
        else:
            grid[x][y] *= 1 + value
        return grid
    for i in range(len(line_points) - 1):
        x0, y0 = line_points[i]
        x1, y1 = line_points[i + 1]
        if abs(x1 - x0) > 90 or abs(y1 - y0) > 90:
            continue
        points = bresenham_line(x0, y0, x1, y1)
        for x, y in points:
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                if modification_method == "solute":
                    grid[x][y] = value
                elif modification_method == "add":
                    grid[x][y] += value
                else:
                    grid[x][y] *= 1 + value

    return grid


def create_predictor_data(
    data_sets,
    n_eofs=5,
    solver_path="solvers/",
):
    """
    Args:
        data_sets (list(str) or str): names of datasets
        n_eofs (int): number of eofs to create for aerosol variables
        solver_path (str): path to save/load EOF solvers
    """

    # Create training and testing arrays
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    X = xr.concat(
        [xr.open_dataset(data_path + f"inputs_{file}.nc") for file in data_sets],
        dim="time",
    )
    X = X.assign_coords(time=np.arange(len(X.time)))

    # Ensure solver_path exists
    os.makedirs(solver_path, exist_ok=True)

    # Function to get or create solver
    def get_or_create_solver(var_name):
        solver_file = os.path.join(solver_path, f"{var_name}_solver.pkl")
        if os.path.exists(solver_file):
            with open(solver_file, "rb") as f:
                solver = pickle.load(f)
        else:
            solver = Eof(X[var_name])
            with open(solver_file, "wb") as f:
                pickle.dump(solver, f)
        return solver

    # Get or create solvers and compute EOFs for BC and SO2
    bc_solver = get_or_create_solver("BC")
    so2_solver = get_or_create_solver("SO2")

    bc_eofs = bc_solver.eofsAsCorrelation(neofs=n_eofs)
    bc_pcs = bc_solver.pcs(npcs=n_eofs, pcscaling=1)

    so2_eofs = so2_solver.eofsAsCorrelation(neofs=n_eofs)
    so2_pcs = so2_solver.pcs(npcs=n_eofs, pcscaling=1)

    # Convert to pandas
    bc_df = bc_pcs.to_dataframe().unstack("mode")
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    so2_df = so2_pcs.to_dataframe().unstack("mode")
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    # Bring the emissions data back together again and normalise
    inputs = pd.DataFrame(
        {"CO2": normalize_co2(X["CO2"].data), "CH4": normalize_ch4(X["CH4"].data)},
        index=X["CO2"].coords["time"].data,
    )

    # Combine with aerosol EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs, (so2_solver, bc_solver)


def get_test_data(day, file, gas, delta, modification_method, modify_points):
    """
    Args:
        file (str): name of the dataset
        n_eofs (int): number of eofs to create for aerosol variables
        solver_path (str): path to load EOF solvers
    """

    solver_path = "solvers/"
    n_eofs = 5
    # Load the input data
    X = xr.open_dataset(data_path + f"inputs_{file}.nc")

    # Function to load a solver from disk
    def load_solver(var_name):
        solver_file = os.path.join(solver_path, f"{var_name}_solver.pkl")
        if os.path.exists(solver_file):
            with open(solver_file, "rb") as f:
                solver = pickle.load(f)
            return solver
        else:
            raise FileNotFoundError(f"Solver for {var_name} not found at {solver_file}")

    # Load the solvers for BC and SO2
    so2_solver = load_solver("SO2")
    bc_solver = load_solver("BC")

    array = X[gas].data[day]
    left_part = array[:, :72]
    right_part = array[:, 72:]
    swapped_array = np.hstack((right_part, left_part))
    processed_array = modify_grid(
        swapped_array, modify_points, modification_method, delta
    )
    final_left_part = processed_array[:, 72:]
    final_right_part = processed_array[:, :72]
    X[gas].data[day] = np.hstack((final_left_part, final_right_part))

    # Project the SO2 and BC fields using the respective solvers
    so2_pcs = so2_solver.projectField(X["SO2"], neofs=n_eofs, eofscaling=1)
    so2_df = so2_pcs.to_dataframe().unstack("mode")
    so2_df.columns = [f"SO2_{i}" for i in range(n_eofs)]

    bc_pcs = bc_solver.projectField(X["BC"], neofs=n_eofs, eofscaling=1)
    bc_df = bc_pcs.to_dataframe().unstack("mode")
    bc_df.columns = [f"BC_{i}" for i in range(n_eofs)]

    # Normalize the CO2 and CH4 data
    inputs = pd.DataFrame(
        {"CO2": normalize_co2(X["CO2"].data), "CH4": normalize_ch4(X["CH4"].data)},
        index=X["CO2"].coords["time"].data,
    )

    # Combine the normalized data with the projected EOFs
    inputs = pd.concat([inputs, bc_df, so2_df], axis=1)
    return inputs


def create_predictdand_data(data_sets):
    if isinstance(data_sets, str):
        data_sets = [data_sets]
    Y = xr.concat(
        [xr.open_dataset(data_path + f"outputs_{file}.nc") for file in data_sets],
        dim="time",
    ).mean("member")
    # Convert the precip values to mm/day
    Y["pr"] *= 86400
    Y["pr90"] *= 86400
    return Y


def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth - pred) ** 2).weighted(weights).mean(["lat", "lon"])).data
