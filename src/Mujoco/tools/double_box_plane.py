import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

XML = """
<mujoco>
    <default>
        <geom solref="-10000 0" friction="{sliding_fric} {torsional_fric} {rolling_fric}"/>
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
    stiffness,
    damping,
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
            stiffness=stiffness,
            damping=damping,
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

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            return_string += f"Time: {(i)/1000} s, Sphere1 Pos: {data.qpos[0]:.2f} {data.qpos[1]:.2f} {data.qpos[2]:.2f}, Vel: {data.qvel[0]:.2f} {data.qvel[1]:.2f} {data.qvel[2]:.2f}\n"
            return_string += f"Time: {(i)/1000} s, Sphere2 Pos: {data.qpos[7]:.2f} {data.qpos[8]:.2f} {data.qpos[9]:.2f}, Vel: {data.qvel[6]:.2f} {data.qvel[7]:.2f} {data.qvel[8]:.2f}\n"

    # 清理资源
    del model
    del data
    mj.set_mjcb_control(None)  # 重置控制回调

    return return_string


if __name__ == "__main__":
    print(
        sphere_collision_simulation(
            stiffness=1000,
            damping=0,
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
