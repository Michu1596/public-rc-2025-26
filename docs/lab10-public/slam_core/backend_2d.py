from typing import List

import numpy as np
from scipy import optimize
import math

from slam_core.models import (
    OdometryMeasurement,
    Point2D,
    Pose2D,
    SensorMeasurement,
)


class SLAM2DBackend:
    """
    The optimization engine.
    """

    def __init__(self):
        pass

    def optimize(
        self,
        initial_poses_guess: List[Pose2D],
        initial_landmarks_guess: List[Point2D],
        measured_movements: List[OdometryMeasurement],
        sensors_measurements: List[SensorMeasurement],
        ground_truth_initial_pose: Pose2D,  # We assume we have the access to the ground truth of starting position
        odometry_noise_std: float = 1.0,
        sensor_noise_std: float = 1.0,
    ) -> optimize.OptimizeResult:
        # Prepare the initial flat parameter vector
        # We optimize over poses and landmarks simuulatenously, so they have to be packed into a single, flat numpy array
        # Parameters: [pose_0, pose_1, ..., pose_N, lm1_x, lm1_y, lm2_x, lm2_y, ...]
        ### TODO ###
        initial_params = np.concatenate([np.array(initial_poses_guess).flatten(), np.array(initial_landmarks_guess).flatten()])
        num_poses = len(initial_poses_guess)
        ### END TODO ###

        # Optimize
        result = optimize.minimize(
            self._objective_function,
            initial_params,
            args=(
                num_poses,
                measured_movements,
                sensors_measurements,
                odometry_noise_std,
                sensor_noise_std,
                ground_truth_initial_pose,
            ),
            method="Powell",
        )

        return result

    def _objective_function(
        self,
        params: np.ndarray,
        num_poses: int,
        measured_movements: List[OdometryMeasurement],
        measurements: List[SensorMeasurement],
        odometry_noise_std: float,
        sensor_noise_std: float,
        ground_truth_initial_pose: Pose2D,
    ) -> float:
        """
        Calculates the error for a given set of parameters (2D pose and landmark estimation).
        """
        # Unpack parameters
        ### TODO ###
        poses = params[:num_poses * 3].reshape((-1,3))
        landmark_coords = params[num_poses * 3:].reshape((-1,2))
        ### END TODO ###

        # Reconstruct landmarks list [(x,y), ...]
        landmarks = []
        for lm in landmark_coords:
            landmarks.append((lm[0], lm[1]))

        # Avoid division by zero
        odom_weight = (
            1.0 / (odometry_noise_std**2) if odometry_noise_std > 1e-9 else 1.0
        )
        sensor_weight = 1.0 / (sensor_noise_std**2) if sensor_noise_std > 1e-9 else 1.0

        # 1. Movement Penalty (Odometry Error)
        # For each odometry measurement, calculate the expected movement based on the current and next pose, and compare to the measured movement.
        ### TODO ###
        calc_movements_deltas = []
        for i in range(len(poses) - 1):
            prev_x, prev_y, prev_theta = poses[i]
            curr_x, curr_y, curr_theta = poses[i + 1]
            calc_movements_deltas.append((curr_x-prev_x, curr_y-prev_y, curr_theta-prev_theta))

        x_movemenents = [(m.delta_x, m.delta_y, m.delta_theta) for m in measured_movements]

        movement_penalty = 0.0
        for i in range(len(poses) - 1):
            x_cal, y_cal, theta_cal = calc_movements_deltas[i]
            x_mes, y_mes, theta_mes = x_movemenents[i]
            movement_penalty += (x_cal - x_mes)**2 + (y_cal - y_mes) ** 2 + (theta_cal - theta_mes) ** 2

        movement_penalty *= odom_weight

        ### END TODO ###

        # 2. Observation Penalty (Sensor Error)
        # For each pose calculate expected distances and angles to all landmarks and compare to measurements
        ### TODO ###


        distance_penalty = 0
        angle_penalty = 0
        for measurement, pose in zip(measurements, poses):
            expected_distances = []
            expected_angles = []
            for lm in landmarks:
                dist = np.sqrt((lm[0] - pose[0]) ** 2 + (lm[1] - pose[1]) ** 2)
                expected_distances.append(dist)
                rel_x = lm[0] - pose[0]
                rel_y = lm[1] - pose[1]
                theta = math.atan2(rel_y, rel_x)
                expected_angles.append(theta - pose[2])  # NOTE idk if i shuld subtract curr angle
            
            diff = np.array(expected_distances) - np.array(
                [dist for dist in measurement.distances]
            )
            diff2 = np.array(expected_angles) - np.array(
                [angle for angle in measurement.angles]
            )
            distance_penalty += np.sum(diff**2)
            angle_penalty += np.sum(diff2**2)

        distance_penalty *= sensor_weight
        angle_penalty *= sensor_weight
        ### END TODO ###

        # 3. Prior (Anchor first pose)
        # This prevents the whole world from shifting arbitrarily
        # We can weight this heavily to ensure it sticks
        (gt_x, gt_y), gt_theta = ground_truth_initial_pose
        prior = (
            (poses[0][0] - gt_x) ** 2
            + (poses[0][1] - gt_y) ** 2
            + (poses[0][2] - gt_theta) ** 2
        ) * 1000.0

        cost = movement_penalty + prior + distance_penalty + angle_penalty
        return cost
