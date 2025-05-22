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

def generate_box_plane_question():
    global questions
    
    # Define ranges for the simulation parameters
    damping_range = (0, 1)
    gra_acel_range = (-10, -9)
    position_range = (-1, 1)
    size_range = (0.05, 0.5)
    mass_range = (0.1, 5)
    velocity_range = (-10, 10)
    simend_range = (1, 5)
    
    # Generate random parameters
    damping = round(random.uniform(*damping_range), 2)
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    x = round(random.uniform(*position_range), 2)
    y = round(random.uniform(*position_range), 2)
    z = round(random.uniform(*position_range), 2)
    r = round(random.uniform(*size_range), 2)
    mass = round(random.uniform(*mass_range), 2)
    x_v = round(random.uniform(*velocity_range), 2)
    y_v = round(random.uniform(*velocity_range), 2)
    z_v = round(random.uniform(*velocity_range), 2)
    simend = random.randint(*simend_range)
    
    # Run simulation
    _, json_data = box_plane_simulation(
        damping=damping,
        gra_acel=gra_acel,
        x=x,
        y=y,
        z=z,
        r=r,
        mass=mass,
        simend=simend,
        x_v=x_v,
        y_v=y_v,
        z_v=z_v
    )
    
    # Create scenario description
    scenario = f"""In a physics experiment, a box is placed on a plane with the following parameters:
    - Damping: {damping}
    - Gravitational acceleration: {gra_acel} m/s²
    - Initial position: ({x}, {y}, {z}) meters
    - Size of the box: {r} meters
    - Mass of the box: {mass} kg
    - Initial velocity: ({x_v}, {y_v}, {z_v}) m/s

The motion of the box is observed for {simend} seconds."""

    # Question about final position in x-direction
    final_position_x = json_data["position"]["x"][-1]
    options, correct_option = generate_number_choice(final_position_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final position of the box in the x-direction after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final position of the box in the x-direction after {simend} seconds is {final_position_x} meters. The answer is {correct_option}."
    })
    
    # Question about final velocity in y-direction
    final_velocity_y = json_data["velocity"]["y"][-1]
    options, correct_option = generate_number_choice(final_velocity_y)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final velocity of the box in the y-direction after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final velocity of the box in the y-direction after {simend} seconds is {final_velocity_y} m/s. The answer is {correct_option}."
    })
    
    # Question about maximum velocity in z-direction
    max_velocity_z = max(abs(v) for v in json_data["velocity"]["z"])
    options, correct_option = generate_number_choice(max_velocity_z)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the maximum absolute velocity of the box in the z-direction during the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The maximum absolute velocity of the box in the z-direction during the {simend}-second observation period is {max_velocity_z} m/s. The answer is {correct_option}."
    })
    
    # Interval question about position in y-direction
    min_pos_y, max_pos_y = min(json_data["position"]["y"]), max(json_data["position"]["y"])
    options, correct_option = generate_interval_choices_(min_pos_y, max_pos_y)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of positions (in meters) that the box occupies in the y-direction during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum position is {min_pos_y} meters and the maximum position is {max_pos_y} meters. The range is {min_pos_y} to {max_pos_y} meters. The answer is {correct_option}."
    })

    # Trend question about velocity in x-direction
    if json_data["velocity"]["x"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity"]["x"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the box's velocity in the x-direction during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The velocity starts at {json_data['velocity']['x'][0]} m/s and ends at {json_data['velocity']['x'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    # Trend question about position in z-direction
    if json_data["position"]["z"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["position"]["z"])
        questions.append({
            "Question": f"{scenario}\n\nHow does the position of the box change in the z-direction over the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The position starts at {json_data['position']['z'][0]} meters and ends at {json_data['position']['z'][-1]} meters. The change in position is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

def generate(world):
    if world == "box_plane":
        generate_box_plane_question()

if __name__ == "__main__":
    for _ in range(200):  # Generate 100 * 10 = 1000 box-plane questions
        generate("box_plane")
        with open('box_plane_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)