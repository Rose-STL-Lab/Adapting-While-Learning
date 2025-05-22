import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import argparse

XML = """
<mujoco>
	<option gravity = " 0 0 {gra_acel}" integrator="RK4" timestep="0.001"/>
	<worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 10" dir="0 0 -1"/>
		<geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
        <body pos="0 0 0.1">
			<joint type="free"/>
			<geom type="sphere" size=".1" rgba=".9 0 0 1" mass="{mass_ball}"/>
		</body>
        <body pos = "3 0 {half_hight}">
            <joint type="free"/>
		    <geom type="cylinder" size="0.1 {half_hight}" rgba="0 1 0 1" mass="{mass_post}"/>
        </body>
		<body pos = "3 0 {box_hight}">
			<joint type="free"/>
			<geom type="box" size=".1 .1 .1" rgba="0.95 0.95 0 1" mass="{mass_box}"/>
		</body>
	</worldbody>
</mujoco>
"""
simend = 0.1  # simulation time


def create_controller(damping):
    def controller(model, data):
        # Apply damping force
        vx, vy, vz = data.qvel[:3]
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        data.xfrc_applied[1][:3] = -damping * v * np.array([vx, vy, vz])

    return controller


def ball_game_simulation(
    damping, gra_acel, mass_ball, mass_post, mass_box, simend, post_hight, x_v, y_v, z_v
):
    model = mj.MjModel.from_xml_string(
        XML.format(
            gra_acel=gra_acel,
            mass_ball=mass_ball,
            mass_post=mass_post,
            mass_box=mass_box,
            half_hight=post_hight / 2,
            box_hight=post_hight + 0.1,
        )
    )
    data = mj.MjData(model)

    data.qvel[0] = x_v
    data.qvel[1] = y_v
    data.qvel[2] = z_v

    return_string = ""
    json_data = {
        "ball": {"position": {"x": [], "y": [], "z": []}, "velocity": {"x": [], "y": [], "z": []}},
        "post": {"position": {"x": [], "y": [], "z": []}},
        "box": {"position": {"x": [], "y": [], "z": []}},
        "game_over": False,
        "game_over_time": None
    }

    mj.set_mjcb_control(create_controller(damping))

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            return_string += f"Time: {(i)/1000} s, Ball Position: x: {data.qpos[0]:.2f} y: {data.qpos[1]:.2f} z: {data.qpos[2]:.2f}, Ball Velocity: x: {data.qvel[0]:.2f} y: {data.qvel[1]:.2f} z: {data.qvel[2]:.2f}, Post Position: x: {data.qpos[7]:.2f} y: {data.qpos[8]:.2f} z: {data.qpos[9]:.2f}, Box Position: x: {data.qpos[14]:.2f} y: {data.qpos[15]:.2f} z: {data.qpos[16]:.2f}\n"
            json_data["ball"]["position"]["x"].append(round(data.qpos[0], 2))
            json_data["ball"]["position"]["y"].append(round(data.qpos[1], 2))
            json_data["ball"]["position"]["z"].append(round(data.qpos[2], 2))
            json_data["ball"]["velocity"]["x"].append(round(data.qvel[0], 2))
            json_data["ball"]["velocity"]["y"].append(round(data.qvel[1], 2))
            json_data["ball"]["velocity"]["z"].append(round(data.qvel[2], 2))
            json_data["post"]["position"]["x"].append(round(data.qpos[7], 2))
            json_data["post"]["position"]["y"].append(round(data.qpos[8], 2))
            json_data["post"]["position"]["z"].append(round(data.qpos[9], 2))
            json_data["box"]["position"]["x"].append(round(data.qpos[14], 2))
            json_data["box"]["position"]["y"].append(round(data.qpos[15], 2))
            json_data["box"]["position"]["z"].append(round(data.qpos[16], 2))

            if data.qpos[16] <= 0.12:
                return_string += "Box was hit by the ball, Game Over!\n"
                json_data["game_over"] = True
                json_data["game_over_time"] = round(i * 0.001, 3)
                break

    del model
    del data
    mj.set_mjcb_control(None)  # 重置控制回调

    return return_string, json_data

def main():
    default_params = {
        "damping": 0,
        "gra_acel": -9.81,
        "mass_ball": 1.0,
        "mass_post": 5.0,
        "mass_box": 1.0,
        "simend": 5,
        "post_hight": 2.0,
        "x_v": 10.0,
        "y_v": 0.0,
        "z_v": 5.0
    }

    print("Running simulation with default parameters:")
    result = ball_game_simulation(**default_params)
    print(result)

    modified_params = default_params.copy()
    modified_params["damping"] = 0
    modified_params["x_v"] = 20.0
    modified_params["z_v"] = 8.0

    print("\nRunning simulation with modified parameters:")
    result = ball_game_simulation(**modified_params)
    print(result)

if __name__ == "__main__":
    main()