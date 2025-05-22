import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import argparse

XML = """
<mujoco>
  <default>
      <geom friction="{sliding_fric} {torsional_fric} {rolling_fric}"/>
  </default>
  <option gravity = " 0 0 {gra_acel}" integrator="RK4" timestep="0.001"/>
  <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="100 100 0.1" rgba=".9 0 0 1"/>
    <body pos="{x} {y} {z}">
			<joint type="free"/>
			<geom type="sphere" size="{r}" rgba=".9 0 0 1" mass="{mass}"/>
		</body>
  </worldbody>
</mujoco>
"""


def create_controller(damping):
    def controller(model, data):
        # Apply damping force
        vx, vy, vz = data.qvel[:3]
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        data.xfrc_applied[1][:3] = -damping * v * np.array([vx, vy, vz])

    return controller


def ball_plane_simulation(
    damping,
    sliding_fric,
    torsional_fric,
    rolling_fric,
    gra_acel,
    x,
    y,
    z,
    r,
    mass,
    simend,
    x_v,
    y_v,
    z_v,
):
    # MuJoCo data structures
    model = mj.MjModel.from_xml_string(
        XML.format(
            gra_acel=gra_acel,
            x=x,
            y=y,
            z=z,
            r=r,
            mass=mass,
            sliding_fric=sliding_fric,
            torsional_fric=torsional_fric,
            rolling_fric=rolling_fric,
        )
    )  # MuJoCo model

    data = mj.MjData(model)  # MuJoCo data

    # Set initial velocity of ball
    data.qvel[0] = x_v
    data.qvel[1] = y_v
    data.qvel[2] = z_v

    return_string = ""
    json_data = {
        "time": [],
        "position": {"x": [], "y": [], "z": []},
        "velocity": {"x": [], "y": [], "z": []}
    }

    mj.set_mjcb_control(create_controller(damping))

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            return_string += f"Time: {(i)/1000} s, Position: x: {data.qpos[0]:.2f} y: {data.qpos[1]:.2f} z: {data.qpos[2]:.2f}, Velocity: x: {data.qvel[0]:.2f} y: {data.qvel[1]:.2f} z: {data.qvel[2]:.2f}\n"
            json_data["position"]["x"].append(round(data.qpos[0], 2))
            json_data["position"]["y"].append(round(data.qpos[1], 2))
            json_data["position"]["z"].append(round(data.qpos[2], 2))
            json_data["velocity"]["x"].append(round(data.qvel[0], 2))
            json_data["velocity"]["y"].append(round(data.qvel[1], 2))
            json_data["velocity"]["z"].append(round(data.qvel[2], 2))

    # 清理资源
    del model
    del data
    mj.set_mjcb_control(None)  # 重置控制回调

    return return_string, json_data


if __name__ == "__main__":
    print(
        ball_plane_simulation(
            damping = 0,
            sliding_fric=0.5,
            torsional_fric=0.5,
            rolling_fric=0.5,
            gra_acel=-9.81,
            x=0,
            y=0,
            z=5,
            r=1,
            mass=0.1,
            simend=1,
            x_v=4,
            y_v=0,
            z_v=0,
        )
    )
    print("Simulation done!")
