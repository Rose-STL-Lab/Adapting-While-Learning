import numpy as np
import random
import json
from utils import *
import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulators.mujoco import *
import random

questions = []

def generate_double_pendulum_question():
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
    mass_capsule_1 = round(random.uniform(*mass_capsule_range), 2)
    mass_ball_1 = round(random.uniform(*mass_ball_range), 2)
    mass_capsule_2 = round(random.uniform(*mass_capsule_range), 2)
    mass_ball_2 = round(random.uniform(*mass_ball_range), 2)
    sliding_fric = round(random.uniform(*friction_range), 2)
    torsional_fric = round(random.uniform(*friction_range), 2)
    rolling_fric = round(random.uniform(*friction_range), 2)
    initial_angle_1 = round(random.uniform(*initial_angle_range), 2)
    initial_angular_velocity_1 = round(random.uniform(*initial_angular_velocity_range), 2)
    initial_angle_2 = round(random.uniform(*initial_angle_range), 2)
    initial_angular_velocity_2 = round(random.uniform(*initial_angular_velocity_range), 2)
    simend = random.randint(*simend_range)  # Random simulation end time
    
    # Run simulation
    _, json_data = double_pendulum_simulation(
        gra_acel=gra_acel,
        mass_capsule_1=mass_capsule_1,
        mass_ball_1=mass_ball_1,
        mass_capsule_2=mass_capsule_2,
        mass_ball_2=mass_ball_2,
        sliding_fric=sliding_fric,
        torsional_fric=torsional_fric,
        rolling_fric=rolling_fric,
        simend=simend,
        initial_angle_1=initial_angle_1,
        initial_angular_velocity_1=initial_angular_velocity_1,
        initial_angle_2=initial_angle_2,
        initial_angular_velocity_2=initial_angular_velocity_2
    )
    
    # Create scenario description
    scenario = f"""In a physics laboratory, a double pendulum experiment is set up with the following parameters:
    - Gravitational acceleration: {gra_acel} m/s²
    - Mass of first pendulum rod: {mass_capsule_1} kg
    - Mass of first pendulum bob: {mass_ball_1} kg
    - Mass of second pendulum rod: {mass_capsule_2} kg
    - Mass of second pendulum bob: {mass_ball_2} kg
    - Sliding friction coefficient: {sliding_fric}
    - Torsional friction coefficient: {torsional_fric}
    - Rolling friction coefficient: {rolling_fric}
    - Initial angle of first pendulum: {initial_angle_1} radians
    - Initial angular velocity of first pendulum: {initial_angular_velocity_1} rad/s
    - Initial angle of second pendulum: {initial_angle_2} radians
    - Initial angular velocity of second pendulum: {initial_angular_velocity_2} rad/s

The pendulum is released and its motion is observed for {simend} seconds."""

    # Final position questions
    final_position_1 = json_data["position_1"][-1]
    options, correct_option = generate_number_choice(final_position_1)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final position of the first pendulum after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final position of the first pendulum after {simend} seconds is {final_position_1} rad. The answer is {correct_option}."
    })
    
    final_position_2 = json_data["position_2"][-1]
    options, correct_option = generate_number_choice(final_position_2)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final position of the second pendulum after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final position of the second pendulum after {simend} seconds is {final_position_2} rad. The answer is {correct_option}."
    })
    
    # Final velocity questions
    final_velocity_1 = json_data["velocity_1"][-1]
    options, correct_option = generate_number_choice(final_velocity_1)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final velocity of the first pendulum after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final velocity of the first pendulum after {simend} seconds is {final_velocity_1} rad/s. The answer is {correct_option}."
    })
    
    final_velocity_2 = json_data["velocity_2"][-1]
    options, correct_option = generate_number_choice(final_velocity_2)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final velocity of the second pendulum after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final velocity of the second pendulum after {simend} seconds is {final_velocity_2} rad/s. The answer is {correct_option}."
    })
    
    # Interval questions about position
    min_pos_1, max_pos_1 = min(json_data["position_1"]), max(json_data["position_1"])
    options, correct_option = generate_interval_choices_(min_pos_1, max_pos_1)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of positions (in radians) that the first pendulum occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum position of the first pendulum is {min_pos_1} rad and the maximum position is {max_pos_1} rad. The range is {min_pos_1} to {max_pos_1} rad. The answer is {correct_option}."
    })
    
    min_pos_2, max_pos_2 = min(json_data["position_2"]), max(json_data["position_2"])
    options, correct_option = generate_interval_choices_(min_pos_2, max_pos_2)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the range of positions (in radians) that the second pendulum occupies during its motion in the {simend}-second observation period?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The minimum position of the second pendulum is {min_pos_2} rad and the maximum position is {max_pos_2} rad. The range is {min_pos_2} to {max_pos_2} rad. The answer is {correct_option}."
    })
    
    # Trend questions about velocity
    if json_data["velocity_1"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity_1"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the first pendulum's velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The velocity of the first pendulum starts at {json_data['velocity_1'][0]} rad/s and ends at {json_data['velocity_1'][-1]} rad/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })
    
    if json_data["velocity_2"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["velocity_2"])
        questions.append({
            "Question": f"{scenario}\n\nWhat is the overall trend of the second pendulum's velocity during the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The velocity of the second pendulum starts at {json_data['velocity_2'][0]} rad/s and ends at {json_data['velocity_2'][-1]} rad/s. The overall trend is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    # Trend questions about position
    if json_data["position_1"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["position_1"])
        questions.append({
            "Question": f"{scenario}\n\nHow does the position of the first pendulum change over the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The position of the first pendulum starts at {json_data['position_1'][0]} rad and ends at {json_data['position_1'][-1]} rad. The change in position is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

    if json_data["position_2"][0] != 0:
        trend_choices, correct_option = generate_trend_quant(json_data["position_2"])
        questions.append({
            "Question": f"{scenario}\n\nHow does the position of the second pendulum change over the {simend}-second observation period?",
            "Options": trend_choices,
            "Correct": correct_option,
            "Solution": f"The position of the second pendulum starts at {json_data['position_2'][0]} rad and ends at {json_data['position_2'][-1]} rad. The change in position is {trend_choices[ord(correct_option)-65]}. The answer is {correct_option}."
        })

def generate(world):
    if world == "double_pendulum":
        generate_double_pendulum_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 double pendulum questions
        generate("double_pendulum")
        with open('double_pendulum_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)