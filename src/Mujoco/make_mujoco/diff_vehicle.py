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

def generate_vehicle_question():
    global questions
    
    # Define ranges for vehicle parameters
    gra_acel_range = (-10, -9)
    friction_range = (0.1, 0.9)
    velocity_range = (-3, 3)
    simend_range = (1, 5)  # New range for simulation end time
    
    # Generate random parameters
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    sliding_fric = round(random.uniform(*friction_range), 2)
    torsional_fric = round(random.uniform(*friction_range), 2)
    rolling_fric = round(random.uniform(*friction_range), 2)
    left_vel = round(random.uniform(*velocity_range), 2)
    right_vel = round(random.uniform(*velocity_range), 2)
    simend = random.randint(*simend_range)  # Random simulation end time
    
    # Run simulation
    _, json_data = diff_vehicle_simulation(
        gra_acel=gra_acel,
        simend=simend,
        left_vel=left_vel,
        right_vel=right_vel,
        sliding_fric=sliding_fric,
        torsional_fric=torsional_fric,
        rolling_fric=rolling_fric
    )
    
    # Create scenario description
    scenario = f"""In a robotics laboratory, a differential drive vehicle experiment is set up with the following parameters:
    - Gravitational acceleration: {gra_acel} m/s²
    - Sliding friction coefficient: {sliding_fric}
    - Torsional friction coefficient: {torsional_fric}
    - Rolling friction coefficient: {rolling_fric}
    - Left wheel velocity: {left_vel} m/s
    - Right wheel velocity: {right_vel} m/s

The vehicle is operated and its motion is observed for {simend} seconds."""

    # Question about final position
    final_position_x = json_data["position"]["x"][-1]
    options, correct_option = generate_number_choice(final_position_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final X position of the vehicle after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final X position of the vehicle after {simend} seconds is {final_position_x} meters. The answer is {correct_option}."
    })

    final_position_y = json_data["position"]["y"][-1]
    options, correct_option = generate_number_choice(final_position_y)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final Y position of the vehicle after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final Y position of the vehicle after {simend} seconds is {final_position_y} meters. The answer is {correct_option}."
    })
    
    # Interval question about position
    min_pos_x, max_pos_x = min(json_data["position"]["x"]), max(json_data["position"]["x"])
    options, correct_option = generate_interval_choices_(min_pos_x, max_pos_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of X positions (in meters) that the vehicle occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum X position is {min_pos_x} meters and the maximum X position is {max_pos_x} meters. The range is {min_pos_x} to {max_pos_x} meters. The answer is {correct_option}."
    })

    # Trend question about velocity
    if json_data["velocity"]["x"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity"]["x"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the vehicle's X velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The X velocity starts at {json_data['velocity']['x'][0]} m/s and ends at {json_data['velocity']['x'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    if json_data["velocity"]["y"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity"]["y"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the vehicle's Y velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The Y velocity starts at {json_data['velocity']['y'][0]} m/s and ends at {json_data['velocity']['y'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

def generate(world):
    if world == "vehicle":
        generate_vehicle_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 vehicle questions
        generate("vehicle")
        with open('diff_vehicle_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)