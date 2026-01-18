import numpy as np
import pandas as pd
import cv2
import math
import tyro
import mujoco
from typing import Optional
from mujoco import viewer
from drone_simulator import DroneSimulator
from pid import PID
from plotting import plot_results
# TODO: Additional imports if needed
from scipy.spatial.transform import Rotation
# END OF TODO
from kalman_filter import KalmanFilter 

# Simulation parameters
resolution = (480, 640)  # (height, width) in pixels
fovy_deg = 90  # vertical field of view in degrees
np.set_printoptions(suppress=True, precision=3)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# TODO: Additional functions if needed

# END OF TODO


def camera_intrinsics_from_fovy(fovy_deg: float, height: int, width: int) -> np.ndarray:
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    K = np.array([[fy, 0,  width / 2.0],
                  [0,  fy, height / 2.0],
                  [0,   0,  1]])
    return K


def update_gate_position(model: mujoco.MjModel, data: mujoco.MjData, gate_name: str, gate_vel: np.ndarray, 
                         gate_motion_prob: float = 0.1,
                         gate_motion_scale: float = 0.15, 
                         gate_noise_scale: float = 0.01, 
                         gate_damping: float = 0.95,
                         sim_dt: Optional[float] = None) -> np.ndarray:
    """Update gate position with smooth motion dynamics."""
    if sim_dt is None:
        sim_dt = model.opt.timestep
    gate_id = model.body(gate_name).id

    # Random impulse
    if np.random.uniform() < gate_motion_prob:
        impulse = np.random.normal(scale=gate_motion_scale, size=3)
        gate_vel += impulse
    
    # Add noise and apply damping
    gate_vel += np.random.normal(scale=gate_noise_scale, size=3)
    gate_vel *= gate_damping
    
    # Integrate to new position
    cur_pos = data.body(gate_name).xpos.copy()
    new_pos = cur_pos + gate_vel * sim_dt * [0.3, 1.0, 1.0]  # slower x motion
    
    # Keep gate within reasonable bounds
    new_pos[0] = np.clip(new_pos[0], -10.0, 2.0)  # x bounds
    new_pos[1] = np.clip(new_pos[1], -1.5, 1.5)   # y bounds
    new_pos[2] = np.clip(new_pos[2], 0.5, 5)      # z bounds
    
    # Apply to model
    model.body_pos[gate_id] = new_pos
    
    return gate_vel


def build_world(rotated_gates: bool) -> str:
    world = open("scene.xml").read()


        # Use random starting positions
    world = world.replace(
        '<body name="red_gate" pos="-2 0 3">',
        f'<body name="red_gate" pos="-2 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(2.7, 3.3)}">'
    )
    
    world = world.replace(
        '<body name="green_gate" pos="-4 -0.6 3.3">',
        f'<body name="green_gate" pos="-4 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(2.7, 3.3)}">'
    )
    world = world.replace(
        '<body name="blue_gate" pos="-6 0.6 2.7">',
        f'<body name="blue_gate" pos="-6 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(2.7, 3.3)}">'
    )

    if rotated_gates:
        world = world.replace(
            '<body name="red_gate"',
            f'<body name="red_gate" euler="0 0 {np.random.uniform(-30, -15) if np.random.rand() < 0.5 else np.random.uniform(15, 30)}"'
        )
        world = world.replace(
            '<body name="green_gate"',
            f'<body name="green_gate" euler="0 0 {np.random.uniform(-30, -15) if np.random.rand() < 0.5 else np.random.uniform(15, 30)}"'
        )
        world = world.replace(
            '<body name="blue_gate"',
            f'<body name="blue_gate" euler="0 0 {np.random.uniform(-30, -15) if np.random.rand() < 0.5 else np.random.uniform(15, 30)}"'
        )
    return world


def run_single_task(*, wind: bool, rotated_gates: bool, flight_mode, rendering_freq: float) -> None:
    world = build_world(rotated_gates)
    model = mujoco.MjModel.from_xml_string(world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = viewer.launch_passive(model, data)
    view.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    view.cam.fixedcamid = model.camera("front_camera").id

    K = camera_intrinsics_from_fovy(fovy_deg, resolution[0], resolution[1])
    dist_coeffs = np.zeros(5)
    desired_thrust = 3.2496
    thrust = desired_thrust
    roll_thrust, pitch_thrust, yaw_thrust = 0.0, 0.0, 0.0

    SIM_TIME = 500 if flight_mode == "hover" else 5000

    pnp_position_list = []  # Store PnP position estimates
    true_position_list = []  # Store true gate positions
    kalman_position_list = []  # Store KF filtered positions
    # TODO: Additional variables if needed
    pnp_position = np.zeros(3)
    kalman_position = np.zeros(3)
    current_marker = 0
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # markers sets
    markers = [
        [0, 1, 2, 3],  # red gate
        [4, 5, 6, 7],  # green gate
        [8, 9, 10, 11] # blue gate
    ]
    current_gate = 0

    # kalmans
    process_var_translation = 0.1
    measurement_var_translation = 0.1
    kf_translation = KalmanFilter(process_var_translation, measurement_var_translation, use_acceleration=True)

    process_var_rotation = 0.1
    measurement_var_rotation = 0.1
    kf_rotation = KalmanFilter(process_var_rotation, measurement_var_rotation, use_acceleration=False)
    
    # pids

    pid_x_thrust = PID(
        gain_prop=10,
        gain_int=0.5,
        gain_der=20,
        sensor_period=model.opt.timestep,
        output_limits=(-15, 15),
    )

    pid_y_thrust = PID(
        gain_prop=10,
        gain_int=0.5,
        gain_der=20,
        sensor_period=model.opt.timestep,
        output_limits=(-15, 15),
    )

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
        gain_prop=100.0,
        gain_int=20,
        gain_der=50.0,
        sensor_period=model.opt.timestep,
        output_limits=(-100, 100),
    )

    # prev x
    prev_state_position = np.zeros(9)
    prev_state_rotation = np.zeros(9)

    # ids
    linacc_id = model.sensor("body_linacc").id
    gyro_id = model.sensor("body_gyro").id

    # dt
    dt = model.opt.timestep
    # END OF TODO

    task_label = f"rotated={'yes' if rotated_gates else 'no'}, wind={'yes' if wind else 'no'}, flight_mode={flight_mode}"
    print(f"Starting task ({task_label})")

    wind_change_prob = 0.1 if wind else 0

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, view, wind_change_prob = wind_change_prob, rendering_freq = rendering_freq
    )

    # --- initiate motion for all 3 gates ---
    red_gate_vel = np.zeros(3, dtype=float)
    green_gate_vel = np.zeros(3, dtype=float)
    blue_gate_vel = np.zeros(3, dtype=float)
    # ---------------------------------------

    # Create renderer once before the loop
    renderer = mujoco.Renderer(model, resolution[0], resolution[1])

    try:
        for i in range(SIM_TIME):
            # ----- update smooth motion of all 3 gates -----
            # Move red_gate along a square path
            red_gate_vel = update_gate_position(model, data, "red_gate", red_gate_vel)
            green_gate_vel = update_gate_position(model, data, "green_gate", green_gate_vel)
            blue_gate_vel = update_gate_position(model, data, "blue_gate", blue_gate_vel)
            mujoco.mj_forward(model, data)
            # -----------------------------------------------
            
            # Render camera frame
            renderer.update_scene(data, camera="front_camera")
            camera_frame = renderer.render()
            camera_frame = np.asarray(camera_frame, dtype=np.uint8)

            # Get current orientation
            current_orien, _ = drone_simulator.orientation_sensor()

            drone_position = drone_simulator.position_sensor()[0]
            if current_marker == 0:
                gate_position = data.body("red_gate").xpos.copy()
            elif current_marker == 4:
                gate_position = data.body("green_gate").xpos.copy()
            elif current_marker == 8:
                gate_position = data.body("blue_gate").xpos.copy()

            # You can use true_position as ground truth for debugging purposes
            true_position = gate_position - drone_position
            print(f"true_position: {true_position.round(3)}")

            # TODO: Detect, estimate pose, apply Kalman filter
            
            # Detect markers
            corners, ids, _ = detector.detectMarkers(camera_frame.copy())

            id0 = markers[current_gate][0]
            id1 = markers[current_gate][1]
            id2 = markers[current_gate][2]
            id3 = markers[current_gate][3]

            corners_dict = {int(id_val): corner for id_val, corner in zip(ids.flatten(), corners)} if ids is not None else {}
            
            # get corners of the current gate
            marker0 = corners_dict.get(id0)
            marker1 = corners_dict.get(id1)
            marker2 = corners_dict.get(id2)
            marker3 = corners_dict.get(id3)

            # update kalmans
            prev_state_position = kf_translation.x.copy().flatten()
            prev_state_rotation = kf_rotation.x.copy().flatten()

            kf_rotation.predict(dt)
            kf_translation.predict(dt)

            # estimate gate pose relative to the drone if all 4 markers are detected
            if (marker0 is not None and marker1 is not None and
                marker2 is not None and marker3 is not None):
                
                # Each corner from ArUco detector has shape (1, 4, 2) - 4 corners per marker
                # We need the center of each marker for PnP
                gate_corners_2d = np.array([
                    marker0[0].mean(axis=0),   # center of left upper marker
                    marker1[0].mean(axis=0),  # center of right upper marker
                    marker2[0].mean(axis=0),  # center of right lower marker
                    marker3[0].mean(axis=0)    # center of left lower marker
                ], dtype=np.float32)

                # Gate-local marker positions
                gate_local_corners = np.array([
                    [0.01, 0.6, 0.65],      # id0
                    [0.01, -0.6, 0.65],     # id1
                    [0.01, -0.6, -0.65],    # id2
                    [0.01, 0.6, -0.65]      # id3
                ], dtype=np.float32)
                
                # Solve PnP
                success, rvec, tvec = cv2.solvePnP(
                    gate_local_corners,
                    gate_corners_2d,
                    K,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                print(f"orginal tvec: {tvec.flatten().round(3)}")

                if success:
                    # Convert rvec to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    
                    # Transform gate-local corners to world using PnP result
                    gate_corners_world = gate_position + (R @ gate_local_corners.T).T
                    
                    # Use world coordinates for PnP - this way PnP estimates drone position
                    gate_corners_3d = gate_corners_world.astype(np.float32)
                    
                    print(f"gate_corners_2d: {gate_corners_2d.round(1)}")
                    print(f"gate_corners_3d: {gate_corners_3d.round(3)}")
                    
                    print(f"tvec: {tvec.flatten().round(3)}")
                    print(f"rvec: {rvec.flatten().round(3)}")
                    # get acceleration from drone simulator
                    # Get the sensor starting indices

                    # Access all 3 values for each sensor
                    acc = data.sensordata[linacc_id:linacc_id+3]  # [ax, ay, az]

                    print(f"acc: {acc} m/s^2")

                    # update kalmans with measurement
                    pnp_position = -R.T @ tvec
                    pnp_position = pnp_position.flatten()
                    measurement_pos = np.vstack((pnp_position.reshape(3,1), acc.reshape(3,1)))
                    print(f"measurement_pos: {measurement_pos.round(3)}")
                    kf_translation.update(measurement_pos)
                    kalman_position = kf_translation.x.flatten()[:3]


            gyro = data.sensordata[gyro_id:gyro_id+3]    # [roll_rate, pitch_rate, yaw_rate]
            print(f"gyro: {gyro} rad/s")
            measurement_rot = np.array(gyro).reshape(3,1)
            print(f"measurement_rot: {measurement_rot.round(3)}")
            kf_rotation.update(measurement_rot)
            kalman_rotation = kf_rotation.x.flatten()[:3]

            print(f"pnp_position: {pnp_position.round(3)}")
            print(f"kalman_position: {kf_translation.x.flatten().round(3)}")
            print(f"kalman_rotation: {kf_rotation.x.flatten().round(3)}")

            # END OF TODO

            # TODO make this fly
            # we want this drone to be in 0,0,0 in gate frame
            desired_position = np.array([0.0, 0.0,0.2])
            target_x = desired_position[0]
            target_y = desired_position[1]

            prev_x = prev_state_position[0]
            current_x = kalman_position[0]

            prev_y = prev_state_position[1]
            current_y = kalman_position[1]

            prev_z = prev_state_position[2]
            current_z = kalman_position[2]

            prev_roll = prev_state_rotation[0]
            current_roll = kalman_rotation[0]

            prev_pitch = prev_state_rotation[1]
            current_pitch = kalman_rotation[1]

            prev_yaw = prev_state_rotation[2]
            current_yaw = kalman_rotation[2]

            desired_x_thrust = pid_x_thrust.output_signal(target_x, [current_x, prev_x])
            desired_y_thrust = pid_y_thrust.output_signal(target_y, [current_y, prev_y])

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

            # roll_thrust = - pid_roll.output_signal(
            #     desired_roll, [current_roll, prev_roll]
            # )
            # pitch_thrust = - pid_pitch.output_signal(
            #     desired_pitch, [current_pitch, prev_pitch]
            # )

            # thrust = pid_z.output_signal(
            #     commanded_variable=desired_position[2],
            #     sensor_readings=[current_z, prev_z],
            # )
            print(f"thrust: {thrust} thrust roll : {roll_thrust} pitch: {pitch_thrust} current z: {current_z.round(3)} prev z: {prev_z.round(3)}")


            # END OF TODO

            if pnp_position is not None:
                pnp_position_list.append(pnp_position.copy())
                kalman_position_list.append(kalman_position.copy())
                true_position_list.append(true_position.copy())


            # Make a simulation step
            drone_simulator.sim_step(
                thrust, roll_thrust,
                pitch_thrust, yaw_thrust
            )

        # Plot the results
        rotated_str = "rotated" if rotated_gates else "straight"
        mode_str = "flight" if flight_mode == "flight" else "hover"
        filename = f"plot_{rotated_str}_{mode_str}.png"
        plot_results(pnp_position_list, true_position_list, kalman_position_list, filename=filename)

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
    runs: int = 1,
    rendering_freq: float = 1
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
            (False, False, "hover"),
            (False, True, "hover"),
        ]
    else:
        task_list = [(False, False, "hover")] # the easiest setup

    for wind_flag, rotated, flight_mode in task_list:
        for run_idx in range(runs):
            print(f"\nRun {run_idx + 1}/{runs} for wind={wind_flag}, rotated_gates={rotated}, flight_mode={flight_mode}")
            run_single_task(
                wind=wind_flag,
                rotated_gates=rotated,
                flight_mode=flight_mode,
                rendering_freq=rendering_freq
            )


if __name__ == '__main__':
    tyro.cli(main)
