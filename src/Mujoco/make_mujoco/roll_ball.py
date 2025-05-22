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

def generate_ball_simulation_question():
    global questions
    
    # Define ranges for ball simulation parameters
    gra_acel_range = (-10, -9)
    pos_range = (-1, 1)
    radius_range = (0.05, 0.2)
    mass_range = (0.1, 5.0)
    friction_range = (0.1, 0.9)
    velocity_range = (-5, 5)
    angular_velocity_range = (-5, 5)
    damping_range = (0, 1)
    simend_range = (1, 5)
    
    # Generate random parameters
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    x = round(random.uniform(*pos_range), 2)
    y = round(random.uniform(*pos_range), 2)
    z = round(random.uniform(*pos_range), 2)
    r = round(random.uniform(*radius_range), 2)
    mass = round(random.uniform(*mass_range), 2)
    sliding_fric = round(random.uniform(*friction_range), 2)
    torsional_fric = round(random.uniform(*friction_range), 2)
    rolling_fric = round(random.uniform(*friction_range), 2)
    x_v = round(random.uniform(*velocity_range), 2)
    y_anv = round(random.uniform(*angular_velocity_range), 2)
    z_v = round(random.uniform(*velocity_range), 2)
    damping = round(random.uniform(*damping_range), 2)
    simend = random.randint(*simend_range)
    
    # Run simulation
    _, json_data = roll_ball_simulation(
        damping=damping,
        gra_acel=gra_acel,
        x=x,
        y=y,
        z=z,
        r=r,
        mass=mass,
        simend=simend,
        x_v=x_v,
        y_anv=y_anv,
        z_v=z_v,
        sliding_fric=sliding_fric,
        torsional_fric=torsional_fric,
        rolling_fric=rolling_fric
    )
    
    # Create scenario description
    scenario = f"""In a physics laboratory, a rolling ball experiment is set up with the following parameters:
    - Gravitational acceleration: {gra_acel} m/s²
    - Initial position: ({x}, {y}, {z}) meters
    - Radius of the ball: {r} meters
    - Mass of the ball: {mass} kg
    - Sliding friction coefficient: {sliding_fric}
    - Torsional friction coefficient: {torsional_fric}
    - Rolling friction coefficient: {rolling_fric}
    - Initial velocity: {x_v} m/s (X), {z_v} m/s (Z)
    - Initial angular velocity: {y_anv} rad/s (Y)
    - Damping coefficient: {damping}
    
The ball is rolled and its motion is observed for {simend} seconds."""

    # Question about final X position
    final_position_x = json_data["position_x"][-1]
    options, correct_option = generate_number_choice(final_position_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final X position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final X position of the ball after {simend} seconds is {final_position_x} meters. The answer is {correct_option}."
    })
    
    # Question about final X velocity
    final_velocity_x = json_data["velocity_x"][-1]
    options, correct_option = generate_number_choice(final_velocity_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final X velocity of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final X velocity of the ball after {simend} seconds is {final_velocity_x} m/s. The answer is {correct_option}."
    })

    # Interval question about X position
    min_pos_x, max_pos_x = min(json_data["position_x"]), max(json_data["position_x"])
    options, correct_option = generate_interval_choices_(min_pos_x, max_pos_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of X positions (in meters) that the ball occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum X position is {min_pos_x} meters and the maximum X position is {max_pos_x} meters. The range is {min_pos_x} to {max_pos_x} meters. The answer is {correct_option}."
    })

    # Trend question about X velocity
    if json_data["velocity_x"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity_x"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the ball's X velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The X velocity starts at {json_data['velocity_x'][0]} m/s and ends at {json_data['velocity_x'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    # Trend question about Angular velocity
    if json_data["angular_velocity_y"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["angular_velocity_y"])
        questions.append({
            "Question": f"{scenario}\n\nHow does the angular velocity of the ball change over the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The angular velocity starts at {json_data['angular_velocity_y'][0]} rad/s and ends at {json_data['angular_velocity_y'][-1]} rad/s. The change in angular velocity is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

def generate(world):
    if world == "ball_simulation":
        generate_ball_simulation_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 ball simulation questions
        generate("ball_simulation")
        with open('ball_simulation_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)