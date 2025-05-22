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

def generate_ball_question():
    global questions
    
    # Define ranges for ball parameters
    damping_range = (0, 1)
    gra_acel_range = (-10, -9)
    position_range = (-5, 5)
    velocity_range = (-5, 5)
    radius_range = (0.05, 0.2)
    mass_range = (0.05, 0.2)
    friction_range = (0.1, 0.9)
    simend_range = (1, 5)  # New range for simulation end time
    
    # Generate random parameters
    damping = round(random.uniform(*damping_range), 2)
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    x = round(random.uniform(*position_range), 2)
    y = round(random.uniform(*position_range), 2)
    z = round(random.uniform(*position_range), 2)
    x_v = round(random.uniform(*velocity_range), 2)
    y_v = round(random.uniform(*velocity_range), 2)
    z_v = round(random.uniform(*velocity_range), 2)
    r = round(random.uniform(*radius_range), 2)
    mass = round(random.uniform(*mass_range), 2)
    sliding_fric = round(random.uniform(*friction_range), 2)
    torsional_fric = round(random.uniform(*friction_range), 2)
    rolling_fric = round(random.uniform(*friction_range), 2)
    simend = random.randint(*simend_range)  # Random simulation end time
    
    # Run simulation
    _, json_data = ball_simulation(
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
        z_v=z_v,
        sliding_fric=sliding_fric,
        torsional_fric=torsional_fric,
        rolling_fric=rolling_fric
    )
    
    # Create scenario description
    scenario = f"""In a physics experiment, a ball is released with the following parameters:
    - Damping: {damping}
    - Gravitational acceleration: {gra_acel} m/s²
    - Initial position: ({x}, {y}, {z})
    - Initial velocity: ({x_v}, {y_v}, {z_v}) m/s
    - Radius: {r} m
    - Mass: {mass} kg
    - Sliding friction coefficient: {sliding_fric}
    - Torsional friction coefficient: {torsional_fric}
    - Rolling friction coefficient: {rolling_fric}

The ball's motion is observed for {simend} seconds."""

    # Question about final position
    final_position = (json_data["position"]["x"][-1], json_data["position"]["y"][-1], json_data["position"]["z"][-1])
    options, correct_option = generate_number_choice(final_position[2])
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final z-position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final z-position of the ball after {simend} seconds is {final_position[2]} m. The answer is {correct_option}."
    })
    
    final_velocity = (json_data["velocity"]["x"][-1], json_data["velocity"]["y"][-1], json_data["velocity"]["z"][-1])
    options, correct_option = generate_number_choice(final_velocity[2])
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final z-velocity of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final z-velocity of the ball after {simend} seconds is {final_velocity[2]} m/s. The answer is {correct_option}."
    })
    
    # Interval question about z-position
    min_pos, max_pos = min(json_data["position"]["z"]), max(json_data["position"]["z"])
    options, correct_option = generate_interval_choices_(min_pos, max_pos)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of z-positions (in meters) that the ball occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum z-position is {min_pos} m and the maximum z-position is {max_pos} m. The range is {min_pos} to {max_pos} m. The answer is {correct_option}."
    })

    # Trend question about z-velocity
    if json_data["velocity"]["z"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity"]["z"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the ball's z-velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The z-velocity starts at {json_data['velocity']['z'][0]} m/s and ends at {json_data['velocity']['z'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    # Trend question about z-position
    if json_data["position"]["z"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["position"]["z"])
        questions.append({
            "Question": f"{scenario}\n\nHow does the z-position of the ball change over the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The z-position starts at {json_data['position']['z'][0]} m and ends at {json_data['position']['z'][-1]} m. The change in z-position is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

def generate(world):
    if world == "ball":
        generate_ball_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 ball questions
        generate("ball")
        with open('ball_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)