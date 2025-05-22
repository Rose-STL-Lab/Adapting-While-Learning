import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

XML = """
<mujoco>
    <default>
        <geom friction="{sliding_fric} {torsional_fric} {rolling_fric}"/>
    </default>
    <option gravity = " 0 0 {gra_acel}" integrator="RK4" timestep="0.001"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <body name="pendulum1" pos="0 0 2">
            <joint name="hinge1" type="hinge" axis="0 1 0"/>
            <geom name="rod1" type="capsule" size=".05" fromto="0 0 0 0 0 -1" rgba=".5 .5 .5 1" mass="{mass_capsule_1}"/>
            <geom name="mass1" type="sphere" pos="0 0 -1" size=".1" rgba=".9 0 0 1" mass="{mass_ball_1}"/>
            <body name="pendulum2" pos="0 0 -1">
                <joint name="hinge2" type="hinge" axis="0 1 0"/>
                <geom name="rod2" type="capsule" size=".05" fromto="0 0 0 0 0 -1" rgba=".5 .5 .5 1" mass="{mass_capsule_2}"/>
                <geom name="mass2" type="sphere" pos="0 0 -1" size=".1" rgba="0 .9 0 1" mass="{mass_ball_2}"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""


def double_pendulum_simulation(
    gra_acel,
    mass_capsule_1,
    mass_ball_1,
    mass_capsule_2,
    mass_ball_2,
    sliding_fric,
    torsional_fric,
    rolling_fric,
    simend,
    initial_angle_1,
    initial_angular_velocity_1,
    initial_angle_2,
    initial_angular_velocity_2,
):
    model = mj.MjModel.from_xml_string(
        XML.format(
            gra_acel=gra_acel,
            mass_capsule_1=mass_capsule_1,
            mass_ball_1=mass_ball_1,
            mass_capsule_2=mass_capsule_2,
            mass_ball_2=mass_ball_2,
            sliding_fric=sliding_fric,
            torsional_fric=torsional_fric,
            rolling_fric=rolling_fric,
        )
    )  # MuJoCo model
    data = mj.MjData(model)  # MuJoCo data

    data.qpos[0] = initial_angle_1
    data.qvel[0] = initial_angular_velocity_1
    data.qpos[1] = initial_angle_2
    data.qvel[1] = initial_angular_velocity_2

    return_string = ""
    json_data = {
        "position_1": [],
        "velocity_1": [],
        "position_2": [],
        "velocity_2": []
    }

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            return_string += f"Time: {(i)/1000} s, Ball 1 Position: {data.qpos[0]:.2f}, Ball 1 Velocity: {data.qvel[0]:.2f}, Ball 2 Position: {data.qpos[1]:.2f}, Ball 2 Velocity: {data.qvel[1]:.2f}\n"
            json_data["position_1"].append(round(data.qpos[0], 2))
            json_data["velocity_1"].append(round(data.qvel[0], 2))
            json_data["position_2"].append(round(data.qpos[1], 2))
            json_data["velocity_2"].append(round(data.qvel[1], 2))

    del model
    del data
    mj.set_mjcb_control(None)

    return return_string, json_data


if __name__ == "__main__":
    print(
        double_pendulum_simulation(
            gra_acel=-9.81,
            mass_capsule_1=0.1,
            mass_ball_1=0.1,
            mass_capsule_2=0.1,
            mass_ball_2=0.1,
            sliding_fric=0.5,
            torsional_fric=0.5,
            rolling_fric=0.5,
            simend=1,
            initial_angle_1=0,
            initial_angular_velocity_1=0,
            initial_angle_2=0,
            initial_angular_velocity_2=5,
        )
    )
    print("Simulation done!")
