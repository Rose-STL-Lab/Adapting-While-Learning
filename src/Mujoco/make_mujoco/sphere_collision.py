from utils import *
import pandas as pd
import argparse
import json
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulators.mujoco import *
import random

questions = []

def generate_sphere_collision_question():
    global questions
    
    # Define ranges for simulation parameters
    timeconst_range = (0.01, 0.1)
    dampratio_range = (0.5, 2)
    gra_acel_range = (-10, -9)
    position_range = (-1, 1)
    radius_range = (0.05, 0.2)
    mass_range = (0.1, 2)
    velocity_range = (-10, 10)
    simend_range = (1, 5)  # Simulation end time range

    # Generate random parameters
    timeconst = round(random.uniform(*timeconst_range), 2)
    dampratio = round(random.uniform(*dampratio_range), 2)
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    x1, y1, z1 = [round(random.uniform(*position_range), 2) for _ in range(3)]
    r1 = round(random.uniform(*radius_range), 2)
    mass1 = round(random.uniform(*mass_range), 2)
    x2, y2, z2 = [round(random.uniform(*position_range), 2) for _ in range(3)]
    r2 = round(random.uniform(*radius_range), 2)
    mass2 = round(random.uniform(*mass_range), 2)
    x1_v, y1_v, z1_v = [round(random.uniform(*velocity_range), 2) for _ in range(3)]
    x2_v, y2_v, z2_v = [round(random.uniform(*velocity_range), 2) for _ in range(3)]
    simend = random.randint(*simend_range)  # Random simulation end time
    
    # Run simulation
    _, json_data = sphere_collision_simulation(
        timeconst=timeconst,
        dampratio=dampratio,
        gra_acel=gra_acel,
        x1=x1,
        y1=y1,
        z1=z1,
        r1=r1,
        mass1=mass1,
        x2=x2,
        y2=y2,
        z2=z2,
        r2=r2,
        mass2=mass2,
        simend=simend,
        x1_v=x1_v,
        y1_v=y1_v,
        z1_v=z1_v,
        x2_v=x2_v,
        y2_v=y2_v,
        z2_v=z2_v
    )
    
    # Create scenario description
    scenario = f"""In a physics simulation, two spheres are set up with the following parameters:
    - Sphere 1: Position ({x1}, {y1}, {z1}), Radius: {r1} m, Mass: {mass1} kg, Initial Velocity: ({x1_v}, {y1_v}, {z1_v}) m/s
    - Sphere 2: Position ({x2}, {y2}, {z2}), Radius: {r2} m, Mass: {mass2} kg, Initial Velocity: ({x2_v}, {y2_v}, {z2_v}) m/s
    - Friction coefficients (sliding, torsional, rolling): (0, 0, 0)
    - Time constant: {timeconst}, Damping ratio: {dampratio}
    - Gravitational acceleration: {gra_acel} m/s²

The spheres are observed for {simend} seconds."""

    # Question about final position of Sphere 1
    final_position_sphere1 = json_data["sphere1"]["position"]["x"][-1]
    options, correct_option = generate_number_choice(final_position_sphere1)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final x-position of Sphere 1 after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final x-position of Sphere 1 after {simend} seconds is {final_position_sphere1} m. The answer is {correct_option}."
    })
    
    # Question about final velocity of Sphere 1
    final_velocity_sphere1 = json_data["sphere1"]["velocity"]["x"][-1]
    options, correct_option = generate_number_choice(final_velocity_sphere1)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final x-velocity of Sphere 1 after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final x-velocity of Sphere 1 after {simend} seconds is {final_velocity_sphere1} m/s. The answer is {correct_option}."
    })

    # Trend question about x-velocity of Sphere 1
    if json_data["sphere1"]["velocity"]["x"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["sphere1"]["velocity"]["x"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of Sphere 1's x-velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The x-velocity starts at {json_data['sphere1']['velocity']['x'][0]} m/s and ends at {json_data['sphere1']['velocity']['x'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })


def generate(world):
    if world == "sphere_collision":
        generate_sphere_collision_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 sphere collision questions
        generate("sphere_collision")
        with open('sphere_collision_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)