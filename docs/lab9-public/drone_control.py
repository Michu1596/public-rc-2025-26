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
        gain_int=0.01,
        gain_der=0.02,
        sensor_period=model.opt.timestep,
        output_limits=(-100, 100),
    )

    pid_pitch = PID(
        gain_prop=0.1,
        gain_int=0.01,
        gain_der=0.02,
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
        gain_int=0.1,
        gain_der=5.0,
        sensor_period=model.opt.timestep,
        output_limits=(-100, 100),
    )

    pid_xy_length = PID(
        gain_prop=1,
        gain_int=0.10,
        gain_der=4,
        sensor_period=model.opt.timestep,
        output_limits=(-15, 15),
    )

    pid_xy_angle_rad = PID(
        gain_prop=5,
        gain_int=0.1,
        gain_der=4,
        sensor_period=model.opt.timestep,
        output_limits=(-6.28, 6.28),
    )

    pid_y = PID(
        gain_prop=5.0,
        gain_int=0.10,
        gain_der=8.5,
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
        rendering_freq=rendering_freq/1000,
    )

    # TODO: Define additional variables if needed
    previous_length_xy = None
    previous_velocity_trust_xy_angle = None
    previous_velocity_angle = None
    # END OF TODO

    try:
        for _ in range(SIM_TIME):
            current_pos, previous_pos = drone_simulator.position_sensor()
            current_orien, previous_orien = drone_simulator.orientation_sensor()

            if np.linalg.norm(np.array(current_pos) - np.array(pos_targets[-1])) < 0.2:
                break

            # TODO: define the current target position
            pos_target = pos_targets[0].copy()

            target_x = 1
            target_y = 1
            # END OF TODO

            # TODO: use PID controllers to steer the drone
            # desired_thrust = 3.2496
            current_x = current_pos[0]
            current_y = current_pos[1]
            current_z = current_pos[2]

            current_roll = current_orien[0]
            current_pitch = current_orien[1]
            current_yaw = current_orien[2]

            desired_thrust = pid_z.output_signal(
                pos_target[2], [current_pos[2], previous_pos[2]]
            )

            # calculate desired yaw angle to face the target
            delta_x = target_x - current_pos[0]
            delta_y = target_y - current_pos[1]
            lenght_xy = math.sqrt(delta_x**2 + delta_y**2)
            if lenght_xy > 0.1:
                desired_yaw = math.degrees(
                    math.atan2(delta_y, delta_x)
                )  # TODO incorporate vector length to avoid jittering at close distances
            print(f"Desired yaw: {desired_yaw}")

            # calculatre desired roll and pitch to move towards the target
            # desired_roll = - pid_y.output_signal( - target_y, [current_pos[1], previous_pos[1]])
            # desired_pitch = pid_x.output_signal( - target_x, [current_pos[0], previous_pos[0]])
            # print(f"Desired pitch: {desired_pitch}")

            # Based on current position and target position, calculate desired roll and pitch factors
            nose_to_target_angle_rad = math.radians(desired_yaw - current_yaw)
            xy_velocity_vector = np.array(
                [current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1]]
            )
            velocity_angle = (
                math.atan2(xy_velocity_vector[1], xy_velocity_vector[0])
                if np.linalg.norm(xy_velocity_vector) > 0.0001
                else 0
            )
            desired_trust_angle = pid_xy_angle_rad.output_signal(
                nose_to_target_angle_rad,
                [
                    velocity_angle,
                    previous_velocity_angle
                    if previous_velocity_angle is not None
                    else velocity_angle,
                ],
            )
            previous_velocity_angle = velocity_angle
            print(f"desired trust angle: {desired_trust_angle} Nose to target angle: {nose_to_target_angle_rad} Current velocity angle: {velocity_angle}")

            pitch_factor = math.cos(desired_trust_angle)
            roll_factor = math.sin(desired_trust_angle)

            # calculate signal based on distance to target in XY plane
            if previous_length_xy is None:
                previous_length_xy = lenght_xy
            signal_xy = pid_xy_length.output_signal(0, [lenght_xy, previous_length_xy])
            previous_length_xy = lenght_xy

            desired_pitch = -pitch_factor * signal_xy
            desired_roll = roll_factor * signal_xy
            print(f"Desired roll: {desired_roll}, Desired pitch: {desired_pitch}")

            roll_thrust = -pid_roll.output_signal(
                desired_roll, [current_orien[0], previous_orien[0]]
            )
            pitch_thrust = -pid_pitch.output_signal(
                desired_pitch, [current_orien[1], previous_orien[1]]
            )
            yaw_thrust = pid_yaw.output_signal(
                desired_yaw * 0 + 90, [current_orien[2], previous_orien[2]]
            )
            print(
                f"Roll thrust: {roll_thrust}, Pitch thrust: {pitch_thrust}, Yaw thrust: {yaw_thrust} Trust: {desired_thrust}"
            )
            print(
                f"Current orientation: Roll: {current_orien[0]}, Pitch: {current_orien[1]}, Yaw: {current_orien[2]}"
            )
            # END OF TODO

            # For debugging purposes you can uncomment, but keep in mind that this slows down the simulation

            data = np.array(
                [
                    pos_target + [desired_roll, desired_pitch, desired_yaw],
                    np.concat([current_pos, current_orien]),
                ]
            ).T
            row_names = ["x", "y", "z", "roll", "pitch", "yaw"]
            headers = ["desired", "current"]
            print(pd.DataFrame(data, index=row_names, columns=headers))

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
    all_tasks: bool = False,
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
