import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import numpy as np
from scipy import stats
from utils.make_problem_utils import *
from tools.diffusion import DiffusionSolver

CACHE_FILE = "simulation_cache.json"

with open("diffusion_questions.json", "r") as f:
    questions = json.load(f)


def generate_question(question, answer):
    question_type = random.choice(["number", "interval"])
    if question_type == "number":
        choices, correct = generate_number_choice(answer)
        questions.append(
            {
                "question": question,
                "options": choices,
                "correct": correct,
            }
        )
    else:
        choices, correct = generate_interval_choices(answer)
        questions.append(
            {
                "question": question,
                "options": choices,
                "correct": correct,
            }
        )


def generate_diffusion_question():
    scenario = random.choice(
        ["heat_conduction", "chemical_diffusion", "population_spread"]
    )

    if scenario == "heat_conduction":
        D = round(random.uniform(0.1, 1.0), 2)
        L = random.randint(10, 100)
        T = random.randint(60, 600)
        unit = "°C"
        variable = "temperature"
        time_unit = "seconds"
        length_unit = "cm"
    elif scenario == "chemical_diffusion":
        D = round(random.uniform(0.0001, 0.001), 4)
        L = random.randint(1, 10)
        T = random.randint(10, 1000)
        unit = "mol/L"
        variable = "concentration"
        time_unit = "seconds"
        length_unit = "cm"
    else:
        D = round(random.uniform(0.1, 1.0), 2)
        L = random.randint(10, 100)
        T = random.randint(1, 10)
        unit = "individuals/km²"
        variable = "population density"
        time_unit = "years"
        length_unit = "km"

    dimension = random.choice(["1D", "2D"])

    solver = DiffusionSolver()

    if dimension == "1D":
        ic_type = random.choice(["constant", "step"])
        if ic_type == "constant":
            initial_value = random.randint(0, 100)
            ic_params = {"value": initial_value}
            ic_description = f"constant initial {variable} of {initial_value} {unit}"
        else:
            left_val = random.randint(0, 100)
            right_val = random.randint(0, 100)
            ic_params = {"left_value": left_val, "right_value": right_val}
            ic_description = f"step function with {left_val} {unit} for x < L/2 and {right_val} {unit} for x ≥ L/2"

        bc_type = random.choice(["dirichlet", "neumann", "periodic"])
        if bc_type == "dirichlet":
            left_val = random.randint(0, 100)
            right_val = random.randint(0, 100)
            bc_params = {"left_value": left_val, "right_value": right_val}
            bc_description = f"Dirichlet boundary conditions with {left_val} {unit} at x=0 and {right_val} {unit} at x=L"
        elif bc_type == "neumann":
            bc_params = {}
            bc_description = "Neumann boundary conditions (zero flux at boundaries)"
        else:
            bc_params = {}
            bc_description = "Periodic boundary conditions"

        solution_types = [
            "get_max_value",
            "get_min_value",
            "get_mean_value",
            "get_value_at_point",
            "get_max_gradient",
        ]
        for solution_type in solution_types:
            if solution_type == "get_value_at_point":
                point = random.randint(0, L)
                result = solver.setup_and_solve(
                    dimension,
                    D,
                    T,
                    L=L,
                    ic_type=ic_type,
                    ic_params=json.dumps(ic_params),
                    bc_type=bc_type,
                    bc_params=json.dumps(bc_params),
                    solution_type=solution_type,
                    solution_x=point,
                )
                answer = round(float(result.split()[-1]), 2)
                question = f"In a 1D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the {variable} ({unit}) at x={point} {length_unit} after {T} {time_unit}? (D={D}, L={L} {length_unit})"
            else:
                result = solver.setup_and_solve(
                    dimension,
                    D,
                    T,
                    L=L,
                    ic_type=ic_type,
                    ic_params=json.dumps(ic_params),
                    bc_type=bc_type,
                    bc_params=json.dumps(bc_params),
                    solution_type=solution_type,
                )
                answer = round(float(result.split()[-1]), 2)
                if solution_type == "get_max_value":
                    question = f"In a 1D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the maximum {variable} ({unit}) after {T} {time_unit}? (D={D}, L={L} {length_unit})"
                elif solution_type == "get_min_value":
                    question = f"In a 1D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the minimum {variable} ({unit}) after {T} {time_unit}? (D={D}, L={L} {length_unit})"
                elif solution_type == "get_mean_value":
                    question = f"In a 1D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the average {variable} ({unit}) after {T} {time_unit}? (D={D}, L={L} {length_unit})"
                elif solution_type == "get_max_gradient":
                    question = f"In a 1D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the maximum gradient of {variable} ({unit}) per {length_unit} after {T} {time_unit}? (D={D}, L={L} {length_unit})"
                elif solution_type == "get_equilibrium_time":
                    question = f"In a 1D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, how long (in {time_unit}) does it take to reach equilibrium? (D={D}, L={L} {length_unit})"

            generate_question(question, answer)

    else:  # 2D diffusion
        Ly = random.randint(5, 10)

        ic_type = random.choice(["constant", "checkerboard"])
        if ic_type == "constant":
            initial_value = random.randint(0, 100)
            ic_params = {"value": initial_value}
            ic_description = f"constant initial {variable} of {initial_value} {unit}"
        else:
            ic_params = {
                "value1": 100,
                "value2": 0,
                "cell_size_x": 0.5,
                "cell_size_y": 0.5,
            }
            ic_description = (
                f"checkerboard pattern with alternating 100 {unit} and 0 {unit}"
            )

        bc_type = random.choice(["dirichlet", "neumann", "periodic"])
        if bc_type == "dirichlet":
            left_val = random.randint(0, 100)
            right_val = random.randint(0, 100)
            bottom_val = random.randint(0, 100)
            top_val = random.randint(0, 100)
            bc_params = {
                "left_value": left_val,
                "right_value": right_val,
                "bottom_value": bottom_val,
                "top_value": top_val,
            }
            bc_description = f"Dirichlet boundary conditions with {left_val}, {right_val}, {bottom_val}, and {top_val} {unit} at left, right, bottom, and top boundaries"
        elif bc_type == "neumann":
            bc_params = {}
            bc_description = "Neumann boundary conditions (zero flux at boundaries)"
        else:
            bc_params = {}
            bc_description = "Periodic boundary conditions"

        solution_types = [
            "get_max_value",
            "get_min_value",
            "get_mean_value",
            "get_value_at_point",
            "get_max_gradient",
        ]
        for solution_type in solution_types:
            if solution_type == "get_value_at_point":
                x = random.randint(0, L)
                y = random.randint(0, Ly)
                result = solver.setup_and_solve(
                    dimension,
                    D,
                    T,
                    Lx=L,
                    Ly=Ly,
                    ic_type=ic_type,
                    ic_params=json.dumps(ic_params),
                    bc_type=bc_type,
                    bc_params=json.dumps(bc_params),
                    solution_type=solution_type,
                    solution_x=x,
                    solution_y=y,
                )
                answer = round(float(result.split()[-1]), 2)
                question = f"In a 2D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the {variable} ({unit}) at (x,y)=({x},{y}) {length_unit} after {T} {time_unit}? (D={D}, Lx={L} {length_unit}, Ly={Ly} {length_unit})"
            else:
                result = solver.setup_and_solve(
                    dimension,
                    D,
                    T,
                    Lx=L,
                    Ly=Ly,
                    ic_type=ic_type,
                    ic_params=json.dumps(ic_params),
                    bc_type=bc_type,
                    bc_params=json.dumps(bc_params),
                    solution_type=solution_type,
                )
                answer = round(float(result.split()[-1]), 2)
                if solution_type == "get_max_value":
                    question = f"In a 2D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the maximum {variable} ({unit}) after {T} {time_unit}? (D={D}, Lx={L} {length_unit}, Ly={Ly} {length_unit})"
                elif solution_type == "get_min_value":
                    question = f"In a 2D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the minimum {variable} ({unit}) after {T} {time_unit}? (D={D}, Lx={L} {length_unit}, Ly={Ly} {length_unit})"
                elif solution_type == "get_mean_value":
                    question = f"In a 2D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the average {variable} ({unit}) after {T} {time_unit}? (D={D}, Lx={L} {length_unit}, Ly={Ly} {length_unit})"
                elif solution_type == "get_max_gradient":
                    question = f"In a 2D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, what is the maximum gradient of {variable} ({unit}) per {length_unit} after {T} {time_unit}? (D={D}, Lx={L} {length_unit}, Ly={Ly} {length_unit})"
                elif solution_type == "get_equilibrium_time":
                    question = f"In a 2D {scenario.replace('_', ' ')} process with {ic_description} and {bc_description}, how long (in {time_unit}) does it take to reach equilibrium? (D={D}, Lx={L} {length_unit}, Ly={Ly} {length_unit})"

            generate_question(question, answer)

    return questions


if __name__ == "__main__":
    # with open("diffusion_questions.json", "r") as f:
    #     questions = json.load(f)
    for _ in range(200):
        generate_diffusion_question()
        print(f"Generated {len(questions)} questions so far.")
        with open("diffusion_questions.json", "w") as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)
