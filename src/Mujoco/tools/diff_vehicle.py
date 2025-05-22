import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

XML = """
<mujoco>
	<option gravity = " 0 0 {gra_acel}" integrator="RK4" timestep="0.001"/>
    <default>
        <geom friction="{sliding_fric} {torsional_fric} {rolling_fric}"/>
    </default>
	<worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
		<geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
		<body name = "chassis" pos="0 0 0.2" euler='0 90 0'>
			<joint type="free" frictionloss="0"/>
			<geom type="box" size=".05 .2 .5" rgba=".9 .9 0 1"/>
			<site name="marker" pos = "0 0 0.1" size="0.1" />
			<body name="left-tire" pos="0 0.3 -0.5" euler='90 0 0'>
				<joint name = "left-wheel" type="hinge" axis="0 0 -1"/>
				<geom type="cylinder" size=".2 0.05" rgba="0 .9 0 1"/>
			</body>
			<body name="right-tire" pos="0 -0.3 -0.5" euler='90 0 0'>
				<joint name = "right-wheel" type="hinge" axis="0 0 -1"/>
				<geom type="cylinder" size=".2 0.05" rgba="0 .9 0 1"/>
			</body>
		</body>
	</worldbody>
	<sensor>
		<framepos objtype="site" objname="marker"/>
	</sensor>
	<actuator>
		<velocity name="left-velocity-servo" joint="left-wheel" kv="100"/>
		<velocity name="right-velocity-servo" joint="right-wheel" kv="100"/>
	</actuator>
</mujoco>
"""


def quat2euler(quat_mujoco):
    # mujocoy quat is constant,x,y,z,
    # scipy quaut is x,y,z,constant
    quat_scipy = np.array(
        [quat_mujoco[3], quat_mujoco[0], quat_mujoco[1], quat_mujoco[2]]
    )

    r = R.from_quat(quat_scipy)
    euler = r.as_euler("xyz", degrees=True)

    return euler


def create_controller(left_vel, right_vel):
    def controller(model, data):
        data.ctrl[0] = left_vel
        data.ctrl[1] = right_vel

    return controller


def diff_vehicle_simulation(
    gra_acel,
    simend,
    left_vel,
    right_vel,
    sliding_fric,
    torsional_fric,
    rolling_fric,
):
    model = mj.MjModel.from_xml_string(
        XML.format(
            gra_acel=gra_acel,
            sliding_fric=sliding_fric,
            torsional_fric=torsional_fric,
            rolling_fric=rolling_fric,
        )
    )  # MuJoCo model
    data = mj.MjData(model)  # Set the controller

    mj.set_mjcb_control(create_controller(left_vel, right_vel))

    return_string = ""
    json_data = {
        "position": {"x": [], "y": [], "z": []},
        "velocity": {"x": [], "y": [], "z": []},
        "wheel_velocity": {"left": [], "right": []},
        "orientation": []
    }

    for i in range(simend * 1000 + 1):
        mj.mj_step(model, data)
        if i % 200 == 0:
            orientation = quat2euler(np.array(data.qpos[3:7]))[2]  # Calculate orientation
            return_string += f"Time: {(i)/1000} s, Position: x: {data.qpos[0]:.2f} y: {data.qpos[1]:.2f} z: {data.qpos[2]:.2f}, Velocity: x: {data.qvel[0]:.2f} y: {data.qvel[1]:.2f} z: {data.qvel[2]:.2f}, Vilocity of Wheels: l: {data.ctrl[0]:.2f} r: {data.ctrl[1]:.2f}, Orientation: {orientation}\n"
            json_data["position"]["x"].append(round(data.qpos[0], 2))
            json_data["position"]["y"].append(round(data.qpos[1], 2))
            json_data["position"]["z"].append(round(data.qpos[2], 2))
            json_data["velocity"]["x"].append(round(data.qvel[0], 2))
            json_data["velocity"]["y"].append(round(data.qvel[1], 2))
            json_data["velocity"]["z"].append(round(data.qvel[2], 2))
            json_data["wheel_velocity"]["left"].append(round(data.ctrl[0], 2))
            json_data["wheel_velocity"]["right"].append(round(data.ctrl[1], 2))
            json_data["orientation"].append(round(orientation, 2))

    # 清理资源
    del model
    del data
    mj.set_mjcb_control(None)  # 重置控制回调

    return return_string, json_data


if __name__ == "__main__":
    print(
        diff_vehicle_simulation(
            gra_acel=-9.81,
            simend=2,
            left_vel=2,
            right_vel=1,
            sliding_fric=1,
            torsional_fric=1,
            rolling_fric=1,
        )
    )
    print("Simulation done!")
