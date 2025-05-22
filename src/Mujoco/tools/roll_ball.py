import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

XML = """
<mujoco>
    <default>
        <geom friction="{sliding_fric} {torsional_fric} {rolling_fric}"/>
    </default>
    <option gravity = " 0 0 {gra_acel}" integrator="RK4" timestep="0.0001"/>
    <asset>
        <material name="floor" texture="checks1" texrepeat="2 2" texuniform="true"/>
        <texture name="checks1" builtin="checker" type='2d' width='256' height='256' rgb1="1 1 1" rgb2="0 0 0" />
        <material name="object" texture="checks2" texrepeat="2 2" texuniform="true"/>
        <texture name="checks2" builtin="checker" type='2d' width='256' height='256' rgb1="1 0 0" rgb2="0 1 0" />
    </asset>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="100 100 0.1" material="floor"/>
        <body pos="{x} {y} {z}">
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 0 1"/>
            <joint type="hinge" axis="0 -1 0"/>
            <geom type="sphere" size="{r}" material="object" mass="{mass}"/>
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


def roll_ball_simulation(
    damping,
    gra_acel,
    x,
    y,
    z,
    r,
    mass,
    simend,
    x_v,
    y_anv,
    z_v,
    sliding_fric=0,
    torsional_fric=0,
    rolling_fric=0,
):
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
    )
    data = mj.MjData(model)

    data.qvel[0] = x_v
    data.qvel[1] = z_v
    data.qvel[2] = y_anv

    mj.set_mjcb_control(create_controller(damping))

    return_string = ""
    json_data = {
        "position_x": [],
        "velocity_x": [],
        "angular_velocity_y": []
    }

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            return_string += f"Time: {(i)/1000} s, Position (X): {data.qpos[0]:.2f}, Velocity (v_x, angular_velocity_y): {data.qvel[0]:.2f} {data.qvel[2]:.2f}\n"
            json_data["position_x"].append(round(data.qpos[0], 2))
            json_data["velocity_x"].append(round(data.qvel[0], 2))
            json_data["angular_velocity_y"].append(round(data.qvel[2], 2))

    del model
    del data
    mj.set_mjcb_control(None)

    return return_string, json_data


if __name__ == "__main__":
    print(
        roll_ball_simulation(
            damping=0,
            gra_acel=-9.81,
            x=0,
            y=0,
            z=0.1,
            r=0.1,
            mass=1,
            simend=2,
            x_v=4,
            y_anv=2,
            z_v=0,
        )
    )
    print("Simulation done!")
