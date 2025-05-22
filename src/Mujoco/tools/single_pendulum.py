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
        <body pos="0 0 2">
            <joint type="hinge" axis="0 1 0"/>
            <geom type="capsule" size=".05" fromto="0 0 0 0 0 -1" rgba=".5 .5 .5 1" mass="{mass_capsule}"/>
            <geom type="sphere" pos="0 0 -1" size=".1" rgba=".9 0 0 1" mass="{mass_ball}"/>
        </body>
    </worldbody>
</mujoco>
"""


def single_pendulum_simulation(
    gra_acel,
    mass_capsule,
    mass_ball,
    sliding_fric,
    torsional_fric,
    rolling_fric,
    simend,
    initial_angle,
    initial_angular_velocity,
):
    model = mj.MjModel.from_xml_string(
        XML.format(
            gra_acel=gra_acel,
            mass_capsule=mass_capsule,
            mass_ball=mass_ball,
            sliding_fric=sliding_fric,
            torsional_fric=torsional_fric,
            rolling_fric=rolling_fric,
        )
    )
    data = mj.MjData(model)

    data.qpos[0] = initial_angle
    data.qvel[0] = initial_angular_velocity

    return_string = ""
    json_data = {"velocity": [], "position": []}

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            return_string += f"Time: {(i)/1000} s, Ball Position: {data.qpos[0]:.2f}, Ball Velocity: {data.qvel[0]:.2f}\n"
            json_data["position"].append(round(data.qpos[0], 2))
            json_data["velocity"].append(round(data.qvel[0], 2))

    del model
    del data
    mj.set_mjcb_control(None)

    return return_string, json_data


if __name__ == "__main__":
    print(
        single_pendulum_simulation( 
            gra_acel=-9.81,
            mass_capsule=0.1,
            mass_ball=0.1,
            sliding_fric=0.5,
            torsional_fric=0.5,
            rolling_fric=0.5,
            simend=1,
            initial_angle=3,
            initial_angular_velocity=0,
        )
    )
    print("Simulation done!")
