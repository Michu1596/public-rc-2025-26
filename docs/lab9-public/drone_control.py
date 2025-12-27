import numpy as np
import pandas as pd
import math
import tyro
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
from drone_simulator import DroneSimulator
from pid import PID


SIM_TIME = 5000  # Maximum simulation time in steps


def xquat_to_euler(xquat):
    return R.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]]).as_euler(
        "xyz", degrees=True
    )


def build_world(fixed_track: bool, rotated_gates: bool) -> str:
    world = open("scene.xml").read()
    if not fixed_track:
        world = world.replace(
            '<body name="red_gate" pos="-2 0 1">',
            f'<body name="red_gate" pos="-2 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">',
        )
        world = world.replace(
            '<body name="green_gate" pos="-4 -0.6 1.3">',
            f'<body name="green_gate" pos="-4 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">',
        )
        world = world.replace(
            '<body name="blue_gate" pos="-6 0.6 0.7">',
            f'<body name="blue_gate" pos="-6 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">',
        )

    if rotated_gates:
        world = world.replace(
            '<body name="red_gate"',
            f'<body name="red_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else -15}"',
        )
        world = world.replace(
            '<body name="green_gate"',
            f'<body name="green_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else -30}"',
        )
        world = world.replace(
            '<body name="blue_gate"',
            f'<body name="blue_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else 45}"',
        )
    return world


def run_single_task(
    *, wind: bool, rotated_gates: bool, rendering_freq: float, fixed_track: bool
) -> None:
    world = build_world(fixed_track, rotated_gates)
    model = mujoco.MjModel.from_xml_string(world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = viewer.launch_passive(model, data)
    view.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    view.cam.fixedcamid = model.camera("track").id

    pos_targets = [
        [0, 0, 1],
        data.body("red_gate").xpos.copy().tolist(),
        data.body("green_gate").xpos.copy().tolist(),
        data.body("blue_gate").xpos.copy().tolist(),
        [-8, 0, 1],
    ]

    yaw_quat_targets = [
        [1, 0, 0, 0],
        data.body("red_gate").xquat.copy().tolist(),
        data.body("green_gate").xquat.copy().tolist(),
        data.body("blue_gate").xquat.copy().tolist(),
        [1, 0, 0, 0],
    ]

    yaw_angle_targets = [xquat_to_euler(xquat)[2] for xquat in yaw_quat_targets]

    # TODO: Design PID control
    pid_roll = PID(
        gain_prop=0.1,
        gain_int=0.0,
        gain_der=0.06,
        sensor_period=model.opt.timestep,
        output_limits=(-100, 100),
    )

    pid_pitch = PID(
        gain_prop=0.1,
        gain_int=0.0,
        gain_der=0.06,
        sensor_period=model.opt.timestep,
        output_limits=(-100, 100),
    )

    pid_yaw = PID(
        gain_prop=0.1,
        gain_int=0.1,
        gain_der=0.1,
        sensor_period=model.opt.timestep,
        output_limits=(-100, 100),
    )

    pid_z = PID(
        gain_prop=10.0,
        gain_int=2,
        gain_der=5.0,
        sensor_period=model.opt.timestep,
        output_limits=(-100, 100),
    )

    pid_x_thrust = PID(
        gain_prop=10,
        gain_int=0,
        gain_der=20,
        sensor_period=model.opt.timestep,
        output_limits=(-15, 15),
    )

    pid_y_thrust = PID(
        gain_prop=10,
        gain_int=0,
        gain_der=20,
        sensor_period=model.opt.timestep,
        output_limits=(-15, 15),
    )

    # END OF TODO

    task_label = (
        f"rotated={'yes' if rotated_gates else 'no'}, wind={'yes' if wind else 'no'}"
    )
    print(f"Starting task ({task_label})")
    data.qpos[0:3] = pos_targets[0]
    data.qpos[3:7] = [1, 0, 0, 0]  # no rotation
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    wind_change_prob = 0.1 if wind else 0

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model,
        data,
        view,
        wind_change_prob=wind_change_prob,
        rendering_freq=rendering_freq,
    )

    # TODO: Define additional variables if needed
    desired_yaw = 0
    lenght_xy = None
    i = 0
    step = 0 # from 0 to 2
    desired_yaw = yaw_angle_targets[i]

    target_x = pos_targets[i][0] 
    target_y = pos_targets[i][1] 
    target_z = pos_targets[i][2] 
    # END OF TODO

    try:
        for _ in range(SIM_TIME):
            current_pos, previous_pos = drone_simulator.position_sensor()
            current_orien, previous_orien = drone_simulator.orientation_sensor()

            if np.linalg.norm(np.array(current_pos) - np.array(pos_targets[-1])) < 0.2:
                break

            # TODO: define the current target position
            pos_target = pos_targets[i].copy()
            if lenght_xy is not None and lenght_xy < 0.25 and i <= len(pos_targets) - 1:
                desired_yaw = yaw_angle_targets[i]

                target_x = pos_targets[i][0] 
                target_y = pos_targets[i][1] 
                target_z = pos_targets[i][2] 

                xy_clearance = 0.3
                if step == 0 and i < len(pos_targets) - 1:
                    target_x += xy_clearance * math.cos(math.radians(desired_yaw))
                    target_y += xy_clearance * math.sin(math.radians(desired_yaw))
                    target_z += .12
                    step = 1
                elif step == 1 and i < len(pos_targets) - 1:
                    step = 2
                    target_z += .12
                elif step == 2 and i < len(pos_targets) - 1:
                    i += 1
                    step = 0
                    target_x -= .5 * math.cos(math.radians(desired_yaw))
                    target_y -= .5 * math.sin(math.radians(desired_yaw))
                    target_z += .12
                    
            # END OF TODO

            # TODO: use PID controllers to steer the drone
            # desired_thrust = 3.2496
            current_x = current_pos[0]
            current_y = current_pos[1]
            current_z = current_pos[2]

            previous_x = previous_pos[0]
            previous_y = previous_pos[1]
            previous_z = previous_pos[2]

            current_roll = current_orien[0]
            current_pitch = current_orien[1]
            current_yaw = current_orien[2]

            desired_thrust = pid_z.output_signal(
                target_z, [current_pos[2], previous_pos[2]]
            )

            # calculate desired yaw angle to face the target
            delta_x = target_x - current_pos[0]
            delta_y = target_y - current_pos[1]
            lenght_xy = math.sqrt(delta_x**2 + delta_y**2)

            # calculate desired x thrust and y thrust to move towards the target
            desired_x_thrust = pid_x_thrust.output_signal(target_x, [current_x, previous_x])
            desired_y_thrust = pid_y_thrust.output_signal(target_y, [current_y, previous_y])
            # print(f"Desired x thrust: {desired_x_thrust}, Desired y thrust: {desired_y_thrust}")

            # based on current yaw, convert desired x and y thrust to desired roll and pitch

            current_yaw_rad = math.radians(current_yaw)
            desired_roll = (
                desired_x_thrust * math.sin(current_yaw_rad)
                - desired_y_thrust * math.cos(current_yaw_rad)
            )
            desired_pitch = (
                + desired_x_thrust * math.cos(current_yaw_rad)
                + desired_y_thrust * math.sin(current_yaw_rad)
            )
            # print(f"Desired roll: {desired_roll}, Desired pitch: {desired_pitch}")

            roll_thrust = - pid_roll.output_signal(
                desired_roll, [current_orien[0], previous_orien[0]]
            )
            pitch_thrust = - pid_pitch.output_signal(
                desired_pitch, [current_orien[1], previous_orien[1]]
            )
            yaw_thrust = pid_yaw.output_signal(
                desired_yaw, [current_orien[2], previous_orien[2]]
            )
            # print(
            #     f"Roll thrust: {roll_thrust}, Pitch thrust: {pitch_thrust}, Yaw thrust: {yaw_thrust} Trust: {desired_thrust}"
            # )
            # print(
            #     f"Current orientation: Roll: {current_orien[0]}, Pitch: {current_orien[1]}, Yaw: {current_orien[2]}"
            # )
            # END OF TODO

            # For debugging purposes you can uncomment, but keep in mind that this slows down the simulation

            # data = np.array(
            #     [
            #         pos_target + [desired_roll, desired_pitch, desired_yaw],
            #         np.concat([current_pos, current_orien]),
            #     ]
            # ).T
            # row_names = ["x", "y", "z", "roll", "pitch", "yaw"]
            # headers = ["desired", "current"]
            # print(pd.DataFrame(data, index=row_names, columns=headers))

            drone_simulator.sim_step(
                desired_thrust,
                roll_thrust=roll_thrust,
                pitch_thrust=pitch_thrust,
                yaw_thrust=yaw_thrust,
            )

        current_pos, _ = drone_simulator.position_sensor()
        assert (
            np.linalg.norm(np.array(current_pos) - np.array(pos_targets[-1])) < 0.2
        ), "Drone did not reach the final target!"
        print(f"Task ({task_label}) completed successfully!")
    finally:
        # Ensure viewer is closed before the next run to avoid multiple open windows.
        try:
            view.close()
        except Exception:
            pass


def main(
    wind: bool = False,
    rotated_gates: bool = False,
    all_tasks: bool = True,
    runs: int = 10,
    rendering_freq: float = 3.0,
    fixed_track: bool = False,
) -> None:
    """
    Run the drone control simulation.

    Args:
        wind: Enable wind disturbances.
        rotated_gates: Rotate gates to create the harder variant.
        all_tasks: Run all four combinations of wind/rotated gates.
        runs: How many times to repeat each selected task.
        rendering_freq: Viewer rendering frequency multiplier (lower slows playback).
    """
    task_list = []
    if all_tasks:
        task_list = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    else:
        task_list = [(wind, rotated_gates)]

    for wind_flag, rotated in task_list:
        for run_idx in range(runs):
            print(
                f"\nRun {run_idx + 1}/{runs} for wind={wind_flag}, rotated_gates={rotated}"
            )
            run_single_task(
                wind=wind_flag,
                rotated_gates=rotated,
                rendering_freq=rendering_freq,
                fixed_track=fixed_track,
            )


if __name__ == "__main__":
    tyro.cli(main)
