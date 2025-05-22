functions_mujoco = {
    "ball_simulation": {
        "name": "ball_simulation",
        "description": "Simulates a single ball's motion using MuJoCo physics. There is nothing in the space other than the ball.",
        "parameters": {
            "type": "object",
            "properties": {
                "stiffness": {
                    "type": "number",
                    "description": "Stiffness of the ball (default: 1000)",
                },
                "damping": {
                    "type": "number",
                    "description": "Damping coefficient (default: 0)",
                },
                "gra_acel": {
                    "type": "number",
                    "description": "Gravitational acceleration in m/s^2 (default: -9.8)",
                },
                "x": {
                    "type": "number",
                    "description": "Initial x-position of the ball (default: 0)",
                },
                "y": {
                    "type": "number",
                    "description": "Initial y-position of the ball (default: 0)",
                },
                "z": {
                    "type": "number",
                    "description": "Initial z-position of the ball (default: 5)",
                },
                "r": {
                    "type": "number",
                    "description": "Radius of the ball (default: 0.1)",
                },
                "mass": {
                    "type": "number",
                    "description": "Mass of the ball in kg (default: 1)",
                },
                "simend": {
                    "type": "number",
                    "description": "Simulation end time in seconds (default: 2)",
                },
                "x_v": {
                    "type": "number",
                    "description": "Initial velocity in x-direction (default: 0)",
                },
                "y_v": {
                    "type": "number",
                    "description": "Initial velocity in y-direction (default: 0)",
                },
                "z_v": {
                    "type": "number",
                    "description": "Initial velocity in z-direction (default: 0)",
                },
                "sliding_fric": {
                    "type": "number",
                    "description": "Sliding friction coefficient (default: 0)",
                },
                "torsional_fric": {
                    "type": "number",
                    "description": "Torsional friction coefficient (default: 0)",
                },
                "rolling_fric": {
                    "type": "number",
                    "description": "Rolling friction coefficient (default: 0)",
                },
            },
            "required": [
                "stiffness",
                "damping",
                "gra_acel",
                "x",
                "y",
                "z",
                "r",
                "mass",
                "simend",
                "x_v",
                "y_v",
                "z_v",
                "sliding_fric",
                "torsional_fric",
                "rolling_fric",
            ],
        },
    },
    "ball_game_simulation": {
        "name": "ball_game_simulation",
        "description": "Simulates a ball game using MuJoCo physics. The game consists of a ball, a post, and a box. The ball is released from a place. Your goal is to make the ball hit the box and avoid hitting the post.",
        "parameters": {
            "type": "object",
            "properties": {
                "damping": {
                    "type": "number",
                    "description": "Damping coefficient (default: 10)",
                },
                "gra_acel": {
                    "type": "number",
                    "description": "Gravitational acceleration in m/s^2 (default: -9.8)",
                },
                "mass_ball": {
                    "type": "number",
                    "description": "Mass of the ball in kg (default: 1)",
                },
                "mass_post": {
                    "type": "number",
                    "description": "Mass of the post in kg (default: 1)",
                },
                "mass_box": {
                    "type": "number",
                    "description": "Mass of the box in kg (default: 1)",
                },
                "simend": {
                    "type": "number",
                    "description": "Simulation end time in seconds (default: 2)",
                },
                "post_hight": {
                    "type": "number",
                    "description": "Height of the post in meters (default: 0.5)",
                },
                "x_v": {
                    "type": "number",
                    "description": "Initial velocity in x-direction (default: 4)",
                },
                "y_v": {
                    "type": "number",
                    "description": "Initial velocity in y-direction (default: 0)",
                },
                "z_v": {
                    "type": "number",
                    "description": "Initial velocity in z-direction (default: 0)",
                },
            },
            "requires": [
                "damping",
                "gra_acel",
                "mass_ball",
                "mass_post",
                "mass_box",
                "simend",
                "post_hight",
                "x_v",
                "y_v",
                "z_v",
            ],
        },
    },
    "ball_plane_simulation": {
        "name": "ball_plane_simulation",
        "description": "Simulates a single ball's motion using MuJoCo physics. There is a plane in the space other than the ball.",
        "parameters": {
            "type": "object",
            "properties": {
                "stiffness": {
                    "type": "number",
                    "description": "Stiffness of the ball (default: 100)",
                },
                "damping": {
                    "type": "number",
                    "description": "Damping coefficient (default: 1)",
                },
                "gra_acel": {
                    "type": "number",
                    "description": "Gravitational acceleration in m/s^2 (default: -9.8)",
                },
                "x": {
                    "type": "number",
                    "description": "Initial x-position of the ball (default: 0)",
                },
                "y": {
                    "type": "number",
                    "description": "Initial y-position of the ball (default: 0)",
                },
                "z": {
                    "type": "number",
                    "description": "Initial z-position of the ball (default: 5)",
                },
                "r": {
                    "type": "number",
                    "description": "Radius of the ball (default: 1)",
                },
                "mass": {
                    "type": "number",
                    "description": "Mass of the ball in kg (default: 0.1)",
                },
                "simend": {
                    "type": "number",
                    "description": "Simulation end time in seconds (default: 2)",
                },
                "x_v": {
                    "type": "number",
                    "description": "Initial velocity in x-direction (default: 4)",
                },
                "y_v": {
                    "type": "number",
                    "description": "Initial velocity in y-direction (default: 0)",
                },
                "z_v": {
                    "type": "number",
                    "description": "Initial velocity in z-direction (default: 0)",
                },
                "sliding_fric": {
                    "type": "number",
                    "description": "Sliding friction coefficient (default: 0)",
                },
                "torsional_fric": {
                    "type": "number",
                    "description": "Torsional friction coefficient (default: 0)",
                },
                "rolling_fric": {
                    "type": "number",
                    "description": "Rolling friction coefficient (default: 0)",
                },
            },
            "required": [
                "stiffness",
                "damping",
                "gra_acel",
                "x",
                "y",
                "z",
                "r",
                "mass",
                "simend",
                "x_v",
                "y_v",
                "z_v",
                "sliding_fric",
                "torsional_fric",
                "rolling_fric",
            ],
        },
    },
    "roll_ball_simulation": {
        "name": "roll_ball_simulation",
        "description": "Simulates a ball rolling on a plane using MuJoCo physics.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "Initial x-position of the ball (default: 0)",
                },
                "y": {
                    "type": "number",
                    "description": "Initial y-position of the ball (default: 0)",
                },
                "z": {
                    "type": "number",
                    "description": "Initial z-position of the ball (default: 0.1)",
                },
                "r": {
                    "type": "number",
                    "description": "Radius of the ball (default: 0.1)",
                },
                "mass": {
                    "type": "number",
                    "description": "Mass of the ball in kg (default: 1)",
                },
                "simend": {
                    "type": "number",
                    "description": "Simulation end time in seconds (default: 2)",
                },
                "x_v": {
                    "type": "number",
                    "description": "Initial velocity in x-direction (default: 4)",
                },
                "y_anv": {
                    "type": "number",
                    "description": "Initial angular velocity in y-axis, positive means anticlockwise (default: 200)",
                },
                "z_v": {
                    "type": "number",
                    "description": "Initial velocity in z-direction (default: 0)",
                },
                "sliding_fric": {
                    "type": "number",
                    "description": "Sliding friction coefficient (default: 0)",
                },
                "torsional_fric": {
                    "type": "number",
                    "description": "Torsional friction coefficient (default: 0)",
                },
                "rolling_fric": {
                    "type": "number",
                    "description": "Rolling friction coefficient (default: 0)",
                },
            },
            "required": [
                "x",
                "y",
                "z",
                "r",
                "mass",
                "simend",
                "x_v",
                "y_anv",
                "z_v",
                "sliding_fric",
                "torsional_fric",
                "rolling_fric",
            ],
        },
    },
    "diff_vehicle_simulation": {
        "name": "diff_vehicle_simulation",
        "description": "Simulates a differential drive car using MuJoCo physics.",
        "parameters": {
            "type": "object",
            "properties": {
                "stiffness": {
                    "type": "number",
                    "description": "Stiffness of the car (default: 100)",
                },
                "damping": {
                    "type": "number",
                    "description": "Damping coefficient (default: 1)",
                },
                "gra_acel": {
                    "type": "number",
                    "description": "Gravitational acceleration in m/s^2 (default: -9.8)",
                },
                "simend": {
                    "type": "number",
                    "description": "Simulation end time in seconds (default: 2)",
                },
                "left_vel": {
                    "type": "number",
                    "description": "Initial velocity of the left wheel (default: 0)",
                },
                "right_vel": {
                    "type": "number",
                    "description": "Initial velocity of the right wheel (default: 0)",
                },
                "sliding_fric": {
                    "type": "number",
                    "description": "Sliding friction coefficient (default: 0)",
                },
                "torsional_fric": {
                    "type": "number",
                    "description": "Torsional friction coefficient (default: 0)",
                },
                "rolling_fric": {
                    "type": "number",
                    "description": "Rolling friction coefficient (default: 0)",
                },
            },
            "required": [
                "stiffness",
                "damping",
                "gra_acel",
                "simend",
                "left_vel",
                "right_vel",
                "sliding_fric",
                "torsional_fric",
                "rolling_fric",
            ],
        },
    },
    "single_pendulum_simulation": {
        "name": "single_pendulum_simulation",
        "description": "Simulates a single pendulum using MuJoCo physics.",
        "parameters": {
            "type": "object",
            "properties": {
                "stiffness": {
                    "type": "number",
                    "description": "Stiffness of the pendulum (default: 100)",
                },
                "damping": {
                    "type": "number",
                    "description": "Damping coefficient (default: 1)",
                },
                "gra_acel": {
                    "type": "number",
                    "description": "Gravitational acceleration in m/s^2 (default: -9.8)",
                },
                "mass_capsule": {
                    "type": "number",
                    "description": "Mass of the capsule in kg (default: 0.1)",
                },
                "mass_ball": {
                    "type": "number",
                    "description": "Mass of the ball in kg (default: 0.1)",
                },
                "sliding_fric": {
                    "type": "number",
                    "description": "Sliding friction coefficient (default: 0)",
                },
                "torsional_fric": {
                    "type": "number",
                    "description": "Torsional friction coefficient (default: 0)",
                },
                "rolling_fric": {
                    "type": "number",
                    "description": "Rolling friction coefficient (default: 0)",
                },
                "simend": {
                    "type": "number",
                    "description": "Simulation end time in seconds (default: 2)",
                },
                "initial_angle": {
                    "type": "number",
                    "description": "Initial angle of the pendulum in radians (default: 0)",
                },
                "initial_angular_velocity": {
                    "type": "number",
                    "description": "Initial angular velocity of the pendulum in rad/s (default: 0)",
                },
            },
            "required": [
                "stiffness",
                "damping",
                "gra_acel",
                "mass_capsule",
                "mass_ball",
                "sliding_fric",
                "torsional_fric",
                "rolling_fric",
                "simend",
                "initial_angle",
                "initial_angular_velocity",
            ],
        },
    },
    "double_pendulum_simulation": {
        "name": "double_pendulum_simulation",
        "description": "Simulates a double pendulum using MuJoCo physics.",
        "parameters": {
            "type": "object",
            "properties": {
                "stiffness": {
                    "type": "number",
                    "description": "Stiffness of the pendulum (default: 100)",
                },
                "damping": {
                    "type": "number",
                    "description": "Damping coefficient (default: 1)",
                },
                "gra_acel": {
                    "type": "number",
                    "description": "Gravitational acceleration in m/s^2 (default: -9.8)",
                },
                "mass_capsule_1": {
                    "type": "number",
                    "description": "Mass of the first capsule in kg (default: 0.1)",
                },
                "mass_ball_1": {
                    "type": "number",
                    "description": "Mass of the first ball in kg (default: 0.1)",
                },
                "mass_capsule_2": {
                    "type": "number",
                    "description": "Mass of the second capsule in kg (default: 0.1)",
                },
                "mass_ball_2": {
                    "type": "number",
                    "description": "Mass of the second ball in kg (default: 0.1)",
                },
                "sliding_fric": {
                    "type": "number",
                    "description": "Sliding friction coefficient (default: 0)",
                },
                "torsional_fric": {
                    "type": "number",
                    "description": "Torsional friction coefficient (default: 0)",
                },
                "rolling_fric": {
                    "type": "number",
                    "description": "Rolling friction coefficient (default: 0)",
                },
                "simend": {
                    "type": "number",
                    "description": "Simulation end time in seconds (default: 2)",
                },
                "initial_angle_1": {
                    "type": "number",
                    "description": "Initial angle of the first pendulum in radians (default: 0)",
                },
                "initial_angular_velocity_1": {
                    "type": "number",
                    "description": "Initial angular velocity of the first pendulum in rad/s (default: 0)",
                },
                "initial_angle_2": {
                    "type": "number",
                    "description": "Initial angle of the second pendulum in radians (default: 0)",
                },
                "initial_angular_velocity_2": {
                    "type": "number",
                    "description": "Initial angular velocity of the second pendulum in rad/s (default: 0)",
                },
            },
            "required": [
                "stiffness",
                "damping",
                "gra_acel",
                "mass_capsule_1",
                "mass_ball_1",
                "mass_capsule_2",
                "mass_ball_2",
                "sliding_fric",
                "torsional_fric",
                "rolling_fric",
                "simend",
                "initial_angle_1",
                "initial_angular_velocity_1",
                "initial_angle_2",
                "initial_angular_velocity_2",
            ],
        },
    },
}
