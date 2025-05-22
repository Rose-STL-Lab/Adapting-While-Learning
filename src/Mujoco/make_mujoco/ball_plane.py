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

def generate_ball_plane_question():
    global questions

    # Define ranges for ball-plane parameters
    damping_range = (0.0, 1.0)
    friction_range = (0.1, 0.9)
    gra_acel_range = (-10, -9)
    position_range = (-5.0, 5.0)
    radius_range = (0.1, 2.0)
    mass_range = (0.05, 0.5)
    velocity_range = (-2.0, 2.0)
    simend_range = (1, 5)  # Range for simulation end time

    # Generate random parameters
    damping = round(random.uniform(*damping_range), 2)
    sliding_fric = round(random.uniform(*friction_range), 2)
    torsional_fric = round(random.uniform(*friction_range), 2)
    rolling_fric = round(random.uniform(*friction_range), 2)
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    x = round(random.uniform(*position_range), 2)
    y = round(random.uniform(*position_range), 2)
    z = round(random.uniform(*position_range), 2)
    r = round(random.uniform(*radius_range), 2)
    mass = round(random.uniform(*mass_range), 2)
    x_v = round(random.uniform(*velocity_range), 2)
    y_v = round(random.uniform(*velocity_range), 2)
    z_v = round(random.uniform(*velocity_range), 2)
    simend = random.randint(*simend_range)  # Random simulation end time

    # Run simulation
    _, json_data = ball_plane_simulation(
        damping=damping,
        sliding_fric=sliding_fric,
        torsional_fric=torsional_fric,
        rolling_fric=rolling_fric,
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
    scenario = f"""In a physics laboratory, a ball-plane experiment is set up with the following parameters:
    - Damping: {damping}
    - Gravitational acceleration: {gra_acel} m/s²
    - Initial position: ({x}, {y}, {z}) meters
    - Radius of the ball: {r} meters
    - Mass of the ball: {mass} kg
    - Sliding friction coefficient: {sliding_fric}
    - Torsional friction coefficient: {torsional_fric}
    - Rolling friction coefficient: {rolling_fric}
    - Initial velocity: ({x_v}, {y_v}, {z_v}) m/s

The ball is released and its motion is observed for {simend} seconds."""

    # Question about final position
    final_position = json_data["position"]
    final_position_x = final_position["x"][-1]
    final_position_y = final_position["y"][-1]
    final_position_z = final_position["z"][-1]
    
    options, correct_option = generate_number_choice(final_position_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final x-position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final x-position of the ball after {simend} seconds is {final_position_x} meters. The answer is {correct_option}."
    })
    
    options, correct_option = generate_number_choice(final_position_y)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final y-position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final y-position of the ball after {simend} seconds is {final_position_y} meters. The answer is {correct_option}."
    })
    
    options, correct_option = generate_interval_choices(final_position_z)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final z-position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final z-position of the ball after {simend} seconds is {final_position_z} meters. The answer is {correct_option}."
    })
    
    # Question about final velocity
    final_velocity = json_data["velocity"]
    final_velocity_x = final_velocity["x"][-1]
    final_velocity_y = final_velocity["y"][-1]
    final_velocity_z = final_velocity["z"][-1]
    
    options, correct_option = generate_number_choice(final_velocity_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final x-velocity of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final x-velocity of the ball after {simend} seconds is {final_velocity_x} m/s. The answer is {correct_option}."
    })
    
    options, correct_option = generate_number_choice(final_velocity_y)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final y-velocity of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final y-velocity of the ball after {simend} seconds is {final_velocity_y} m/s. The answer is {correct_option}."
    })
    
    options, correct_option = generate_number_choice(final_velocity_z)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final z-velocity of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final z-velocity of the ball after {simend} seconds is {final_velocity_z} m/s. The answer is {correct_option}."
    })
    
    # Interval question about position
    min_pos_x, max_pos_x = min(json_data["position"]["x"]), max(json_data["position"]["x"])
    options, correct_option = generate_interval_choices_(min_pos_x, max_pos_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of x-positions that the ball occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum x-position is {min_pos_x} meters and the maximum x-position is {max_pos_x} meters. The range is {min_pos_x} to {max_pos_x} meters. The answer is {correct_option}."
    })
    
    min_pos_y, max_pos_y = min(json_data["position"]["y"]), max(json_data["position"]["y"])
    options, correct_option = generate_interval_choices_(min_pos_y, max_pos_y)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of y-positions that the ball occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum y-position is {min_pos_y} meters and the maximum y-position is {max_pos_y} meters. The range is {min_pos_y} to {max_pos_y} meters. The answer is {correct_option}."
    })
    
    min_pos_z, max_pos_z = min(json_data["position"]["z"]), max(json_data["position"]["z"])
    options, correct_option = generate_interval_choices_(min_pos_z, max_pos_z)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of z-positions that the ball occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum z-position is {min_pos_z} meters and the maximum z-position is {max_pos_z} meters. The range is {min_pos_z} to {max_pos_z} meters. The answer is {correct_option}."
    })

    # Trend question about velocity
    if json_data["velocity"]["x"] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity"]["x"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the ball's x-velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The x-velocity starts at {json_data['velocity']['x'][0]} m/s and ends at {json_data['velocity']['x'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    if json_data["velocity"]["z"] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity"]["z"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the ball's z-velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The z-velocity starts at {json_data['velocity']['z'][0]} m/s and ends at {json_data['velocity']['z'][-1]} m/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

def generate(world):
    if world == "ball_plane":
        generate_ball_plane_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 ball-plane questions
        generate("ball_plane")
        with open('ball_plane_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)