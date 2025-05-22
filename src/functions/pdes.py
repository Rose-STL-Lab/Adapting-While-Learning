function_pdes = {
    "setup_and_solve": {
        "name": "setup_and_solve",
        "description": "Set up the PDE problem parameters and solve the diffusion equation.",
        "properties": {
            "type": "object",
            "properties": {
                "dimension": {
                    "type": "string",
                    "enum": ["1D", "2D"],
                    "description": "Dimension of the PDE problem. Can be '1D' or '2D'.",
                },
                "D": {
                    "type": "number",
                    "description": "Diffusion coefficient, a measure of how fast diffusion occurs.",
                },
                "T": {"type": "number", "description": "Final time of the simulation."},
                "L": {
                    "type": "number",
                    "description": "Length of the domain in the x-direction (only for 1D dimension).",
                },
                "Lx": {
                    "type": "number",
                    "description": "Length of the domain in the x-direction (only for 2D diffusion).",
                },
                "Ly": {
                    "type": "number",
                    "description": "Length of the domain in the y-direction (only for 2D diffusion).",
                },
                "ic_type": {
                    "type": "string",
                    "enum": ["constant", "step", "checkerboard"],
                    "description": "Type of the initial condition.",
                },
                "ic_params": {
                    "type": "object",
                    "description": "Parameters for the initial condition, specified as a JSON object. The content depends on the type of initial condition chosen.",
                },
                "bc_type": {
                    "type": "string",
                    "enum": ["dirichlet", "neumann", "periodic"],
                    "description": "Type of the boundary condition.",
                },
                "bc_params": {
                    "type": "object",
                    "description": "Parameters for the boundary condition, specified as a JSON object. The content depends on the type of boundary condition chosen.",
                },
                "solution_type": {
                    "type": "string",
                    "enum": [
                        "get_max_value",
                        "get_min_value",
                        "get_mean_value",
                        "get_value_at_point",
                        "get_max_gradient",
                    ],
                    "description": "Type of solution to retrieve after solving the diffusion equation.",
                },
                "solution_x": {
                    "type": "number",
                    "description": "The x-coordinate of the point to get the value",
                },
                "solution_y": {
                    "type": "number",
                    "description": "The y-coordinate of the point to get the value (only for 2D diffusion).",
                },
            },
            "required": [
                "dimension",
                "D",
                "T",
                "ic_type",
                "ic_params",
                "bc_type",
                "bc_params",
                "solution_type",
            ],
        },
    }
}
