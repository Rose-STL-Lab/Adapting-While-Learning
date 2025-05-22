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

def generate_pendulum_question():
    global questions
    
    # Define ranges for pendulum parameters
    gra_acel_range = (-10, -9)
    mass_capsule_range = (0.05, 0.2)
    mass_ball_range = (0.05, 0.2)
    friction_range = (0.1, 0.9)
    initial_angle_range = (0.1, 3.0)
    initial_angular_velocity_range = (-1, 1)
    simend_range = (1, 5)  # New range for simulation end time
    
    # Generate random parameters
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    mass_capsule = round(random.uniform(*mass_capsule_range), 2)
    mass_ball = round(random.uniform(*mass_ball_range), 2)
    sliding_fric = round(random.uniform(*friction_range), 2)
    torsional_fric = round(random.uniform(*friction_range), 2)
    rolling_fric = round(random.uniform(*friction_range), 2)
    initial_angle = round(random.uniform(*initial_angle_range), 2)
    initial_angular_velocity = round(random.uniform(*initial_angular_velocity_range), 2)
    simend = random.randint(*simend_range)  # Random simulation end time
    
    # Run simulation
    _, json_data = single_pendulum_simulation(
        gra_acel=gra_acel,
        mass_capsule=mass_capsule,
        mass_ball=mass_ball,
        sliding_fric=sliding_fric,
        torsional_fric=torsional_fric,
        rolling_fric=rolling_fric,
        simend=simend,
        initial_angle=initial_angle,
        initial_angular_velocity=initial_angular_velocity
    )
    
    # Create scenario description
    scenario = f"""In a physics laboratory, a simple pendulum experiment is set up with the following parameters:
    - Gravitational acceleration: {gra_acel} m/s²
    - Mass of pendulum rod: {mass_capsule} kg
    - Mass of pendulum bob: {mass_ball} kg
    - Sliding friction coefficient: {sliding_fric}
    - Torsional friction coefficient: {torsional_fric}
    - Rolling friction coefficient: {rolling_fric}
    - Initial angle: {initial_angle} radians
    - Initial angular velocity: {initial_angular_velocity} rad/s

The pendulum is released and its motion is observed for {simend} seconds."""

    # Question about final position
    final_position = json_data["position"][-1]
    options, correct_option = generate_number_choice(final_position)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final position of the pendulum after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final position of the pendulum after {simend} seconds is {final_position} rad. The answer is {correct_option}."
    })
    
    final_velocity = json_data["velocity"][-1]
    options, correct_option = generate_number_choice(final_velocity)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final velocity of the pendulum after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final velocity of the pendulum after {simend} seconds is {final_velocity} rad/s. The answer is {correct_option}."
    })
    
    # Interval question about position
    min_pos, max_pos = min(json_data["position"]), max(json_data["position"])
    options, correct_option = generate_interval_choices_(min_pos, max_pos)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of positions (in radians) that the pendulum occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum position is {min_pos} rad and the maximum position is {max_pos} rad. The range is {min_pos} to {max_pos} rad. The answer is {correct_option}."
    })

    # Trend question about velocity
    if json_data["velocity"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the pendulum's velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The velocity starts at {json_data['velocity'][0]} rad/s and ends at {json_data['velocity'][-1]} rad/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    # Trend question about position
    if json_data["position"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["position"])
        questions.append({
            "Question": f"{scenario}\n\nHow does the position of the pendulum change over the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The position starts at {json_data['position'][0]} rad and ends at {json_data['position'][-1]} rad. The change in position is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

def generate(world):
    if world == "pendulum":
        generate_pendulum_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 pendulum questions
        generate("pendulum")
        with open('single_pendulum_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)