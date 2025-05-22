import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

XML = """
<mujoco>
    <default>
        <geom solref="{timeconst} {dampratio}" friction="{sliding_fric} {torsional_fric} {rolling_fric}"/>
    </default>
    <option gravity="0 0 {gra_acel}" integrator="RK4" timestep="0.0001"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="100 100 0.1"/>
        <body pos="{x1} {y1} {z1}">
            <joint type="free"/>
            <geom type="sphere" size="{r1}" mass="{mass1}"/>
        </body>
        <body pos="{x2} {y2} {z2}">
            <joint type="free"/>
            <geom type="sphere" size="{r2}" mass="{mass2}"/>
        </body>
    </worldbody>
</mujoco>
"""

def sphere_collision_simulation(
    timeconst,
    dampratio,
    gra_acel,
    x1,
    y1,
    z1,
    r1,
    mass1,
    x2,
    y2,
    z2,
    r2,
    mass2,
    simend,
    x1_v,
    y1_v,
    z1_v,
    x2_v,
    y2_v,
    z2_v,
    sliding_fric=0,
    torsional_fric=0,
    rolling_fric=0,
):
    model = mj.MjModel.from_xml_string(
        XML.format(
            timeconst=timeconst,
            dampratio=dampratio,
            gra_acel=gra_acel,
            x1=x1,
            y1=y1,
            z1=z1,
            r1=r1,
            mass1=mass1,
            x2=x2,
            y2=y2,
            z2=z2,
            r2=r2,
            mass2=mass2,
            sliding_fric=sliding_fric,
            torsional_fric=torsional_fric,
            rolling_fric=rolling_fric,
        )
    )
    data = mj.MjData(model)

    data.qvel[0:3] = [x1_v, y1_v, z1_v]
    data.qvel[6:9] = [x2_v, y2_v, z2_v]

    return_string = ""
    json_data = {
        "sphere1": {
            "position": {"x": [], "y": [], "z": []},
            "velocity": {"x": [], "y": [], "z": []}
        },
        "sphere2": {
            "position": {"x": [], "y": [], "z": []},
            "velocity": {"x": [], "y": [], "z": []}
        }
    }

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            return_string += f"Time: {(i)/1000} s, Sphere1 Pos: x: {data.qpos[0]:.2f} y: {data.qpos[1]:.2f} z: {data.qpos[2]:.2f}, Vel: x: {data.qvel[0]:.2f} y: {data.qvel[1]:.2f} z: {data.qvel[2]:.2f}\n"
            return_string += f"Time: {(i)/1000} s, Sphere2 Pos: x: {data.qpos[7]:.2f} y: {data.qpos[8]:.2f} z: {data.qpos[9]:.2f}, Vel: x: {data.qvel[6]:.2f} y: {data.qvel[7]:.2f} z: {data.qvel[8]:.2f}\n"
            json_data["sphere1"]["position"]["x"].append(round(data.qpos[0], 2))
            json_data["sphere1"]["position"]["y"].append(round(data.qpos[1], 2))
            json_data["sphere1"]["position"]["z"].append(round(data.qpos[2], 2))
            json_data["sphere1"]["velocity"]["x"].append(round(data.qvel[0], 2))
            json_data["sphere1"]["velocity"]["y"].append(round(data.qvel[1], 2))
            json_data["sphere1"]["velocity"]["z"].append(round(data.qvel[2], 2))
            json_data["sphere2"]["position"]["x"].append(round(data.qpos[7], 2))
            json_data["sphere2"]["position"]["y"].append(round(data.qpos[8], 2))
            json_data["sphere2"]["position"]["z"].append(round(data.qpos[9], 2))
            json_data["sphere2"]["velocity"]["x"].append(round(data.qvel[6], 2))
            json_data["sphere2"]["velocity"]["y"].append(round(data.qvel[7], 2))
            json_data["sphere2"]["velocity"]["z"].append(round(data.qvel[8], 2))

    del model
    del data
    mj.set_mjcb_control(None) 

    return return_string, json_data


if __name__ == "__main__":
    print(
        sphere_collision_simulation(
            timeconst = 0.02,
            dampratio = 1,
            gra_acel=-9.81,
            x1=0,
            y1=0,
            z1=0.1,
            r1=0.1,
            mass1=1,
            x2=1,
            y2=0,
            z2=0.1,
            r2=0.1,
            mass2=1,
            simend=2,
            x1_v=20,
            y1_v=0,
            z1_v=0,
            x2_v=0,
            y2_v=0,
            z2_v=0,
        )
    )
    print("Simulation done!")
