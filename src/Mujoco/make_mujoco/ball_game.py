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

def generate_ball_game_question():
    global questions
    
    # Define ranges for ball game parameters
    damping_range = (0, 1)
    gra_acel_range = (-10, -9)
    mass_ball_range = (0.5, 2.0)
    mass_post_range = (1.0, 10.0)
    mass_box_range = (0.5, 2.0)
    post_hight_range = (1.0, 5.0)
    velocity_range = (0, 20)
    
    # Generate random parameters
    damping = round(random.uniform(*damping_range), 2)
    gra_acel = round(random.uniform(*gra_acel_range), 2)
    mass_ball = round(random.uniform(*mass_ball_range), 2)
    mass_post = round(random.uniform(*mass_post_range), 2)
    mass_box = round(random.uniform(*mass_box_range), 2)
    post_hight = round(random.uniform(*post_hight_range), 2)
    x_v = round(random.uniform(*velocity_range), 2)
    y_v = round(random.uniform(*velocity_range), 2)
    z_v = round(random.uniform(*velocity_range), 2)
    simend = random.randint(1, 5)  # Random simulation end time

    # Run simulation
    _, json_data = ball_game_simulation(
        damping=damping,
        gra_acel=gra_acel,
        mass_ball=mass_ball,
        mass_post=mass_post,
        mass_box=mass_box,
        simend=simend,
        post_hight=post_hight,
        x_v=x_v,
        y_v=y_v,
        z_v=z_v
    )

    # Create scenario description
    scenario = f"""In a physics experiment, a ball game simulation is set up with the following parameters:
    - Damping coefficient: {damping}
    - Gravitational acceleration: {gra_acel} m/s²
    - Mass of the ball: {mass_ball} kg
    - Mass of the post: {mass_post} kg
    - Mass of the box: {mass_box} kg
    - Height of the post: {post_hight} meters
    - Initial velocity of the ball: ({x_v}, {y_v}, {z_v}) m/s

The ball is released and its motion is observed for {simend} seconds."""

    # Question about final position of ball
    final_position_x = json_data["ball"]["position"]["x"][-1]
    final_position_y = json_data["ball"]["position"]["y"][-1]
    final_position_z = json_data["ball"]["position"]["z"][-1]
    final_position = (final_position_x, final_position_y, final_position_z)

    options, correct_option = generate_number_choice(final_position_x)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final x-position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final x-position of the ball after {simend} seconds is {final_position_x} meters. The answer is {correct_option}."
    })

    options, correct_option = generate_interval_choices(final_position_y)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final y-position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final y-position of the ball after {simend} seconds is {final_position_y} meters. The answer is {correct_option}."
    })

    options, correct_option = generate_number_choice(final_position_z)
    questions.append({
        "Question": f"{scenario}\n\nWhat is the final z-position of the ball after {simend} seconds?",
        "Options": options,
        "Correct": correct_option,
        "Solution": f"The final z-position of the ball after {simend} seconds is {final_position_z} meters. The answer is {correct_option}."
    })


def generate(world):
    if world == "ball_game":
        generate_ball_game_question()

if __name__ == "__main__":
    for _ in range(100):  # Generate 100 * 10 = 1000 ball game questions
        generate("ball_game")
        with open('ball_game_questions.json', 'w') as f:
            json.dump(questions, f, cls=NumpyEncoder, indent=4)