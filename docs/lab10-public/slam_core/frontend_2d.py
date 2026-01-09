import math
import random
from typing import List, Tuple

from .models import OdometryMeasurement, Point2D, Pose2D, SensorMeasurement

class SLAM2DFrontend:
    """
    Simulates the environment and the robot's sensors in 2D world.
    Generates ground truth data and noisy measurements.
    """

    def __init__(
        self,
        initial_pose: Pose2D = ((0, 0), 0),
        trajectory_shape: str = "square",
        num_steps: int = 3,
        odometry_noise_std: float = 0.0,
        sensor_noise_std: float = 0.0,
        initial_guess_std: float = 0.1,
    ):
        self.initial_pose = initial_pose
        self.num_steps = num_steps
        self.trajectory_shape = trajectory_shape
        self.odometry_noise_std = odometry_noise_std
        self.sensor_noise_std = sensor_noise_std
        self.initial_guess_std = initial_guess_std

        self.true_poses: List[Pose2D] = []
        self.true_landmarks: List[Point2D] = []

        # Measurements (potentially noisy)
        self.odometry_measurements: List[OdometryMeasurement] = []
        self.sensor_measurements: List[SensorMeasurement] = []

    def add_landmark(self, x: float, y: float):
        self.true_landmarks.append((x, y))

    def generate_ground_truth(self) -> None:
        """Generates the true trajectory of the robot."""
        if self.trajectory_shape == "square":
            self.true_poses = [self.initial_pose]
            x, y = self.initial_pose[0]
            angle = self.initial_pose[1]
            ### TODO ###
            steps_per_side = 10
            step_size = 1
            for _ in range(4):
                for _ in range(steps_per_side):
                    delta_x = math.cos(angle) * step_size
                    delta_y = math.sin(angle) * step_size
                    x += delta_x
                    y += delta_y
                    self.true_poses.append(((x,y), 0)) # Zero na siłę bo w wizualizacji coś dziwnego się z tym kątem dzieje
                    print(f"Added pose: {self.true_poses[-1]}")
                
                angle += math.pi / 2
            ### END TODO ###
        elif self.trajectory_shape == "circle":
            x, y = self.initial_pose[0]
            angle = self.initial_pose[1]
            ### TODO ###
            steps = 30
            step_size = 1
            for _ in range(steps):
                    delta_x = math.cos(angle) * step_size
                    delta_y = math.sin(angle) * step_size
                    x += delta_x
                    y += delta_y
                    self.true_poses.append(((x,y), 0)) # Zero na siłę bo w wizualizacji coś dziwnego się z tym kątem dzieje
                    angle += math.pi * 2 / steps
            ### END TODO ###
        elif self.trajectory_shape == "line":
            self.true_poses = [self.initial_pose]
            x, y = self.initial_pose[0]
            angle = self.initial_pose[1]
            ### TODO ###
            steps = 30
            step_size = 0.1
            for _ in range(steps):
                    delta_x = math.cos(angle) * step_size
                    delta_y = math.sin(angle) * step_size
                    x += delta_x
                    y += delta_y
                    self.true_poses.append(((x,y), 0)) # Zero na siłę bo w wizualizacji coś dziwnego się z tym kątem dzieje
            ### END TODO ###

    def generate_measurements(self) -> None:
        """
        Generates odometry and sensor measurements based on ground truth,
        applying specified noise.
        """
        if not self.true_poses:
            raise ValueError(
                "Ground truth not generated. Call generate_ground_truth() first."
            )

        # 1. Generate Odometry (Movements)
        # Calculate the robot's true movements in RELATIVE frame of reference
        true_odometry = []
        for i in range(len(self.true_poses) - 1):
            ### TODO ###
            ...
            (prev_x, prev_y), prev_theta = self.true_poses[i]
            (curr_x, curr_y), curr_theta = self.true_poses[i+1]
            delta_x = curr_x - prev_x
            delta_y = curr_y - prev_y
            delta_theta = curr_theta - prev_theta
            ### END TODO ###
            true_odometry.append((delta_x, delta_y, delta_theta))
            print(f"True odometry {i}: {(delta_x, delta_y, delta_theta)}")

        # Add noise to odometry measurements
        for odometry in true_odometry:
            delta_x, delta_y, delta_theta = odometry
            delta_x += random.gauss(0, self.odometry_noise_std)
            delta_y += random.gauss(0, self.odometry_noise_std)
            delta_theta += random.gauss(0, self.odometry_noise_std)
            self.odometry_measurements.append(
                OdometryMeasurement(
                    step_index=len(self.odometry_measurements),
                    delta_x=delta_x,
                    delta_y=delta_y,
                    delta_theta=delta_theta,
                )
            )

        # 2. Generate Sensor Readings (Distances and angles to landmarks)
        true_sensor_readings = []
        for pose in self.true_poses:
            distances, angles = self._calculate_observations_for_pose(
                pose, self.true_landmarks
            )
            true_sensor_readings.append((distances, angles))

        # Add noise to sensor measurements
        self.sensor_measurements = []
        for pose_idx, readings_at_pose in enumerate(true_sensor_readings):
            noisy_distances = [
                dist + random.gauss(0, self.sensor_noise_std)
                for dist in readings_at_pose[0]
            ]
            noisy_angles = [
                angle + random.gauss(0, self.sensor_noise_std)
                for angle in readings_at_pose[1]
            ]
            self.sensor_measurements.append(
                SensorMeasurement(
                    pose_index=pose_idx, distances=noisy_distances, angles=noisy_angles
                )
            )

    @staticmethod
    def _calculate_observations_for_pose(
        current_pose_val: Pose2D, landmarks: List[Point2D]
    ) -> Tuple[List[float], List[float]]:
        """Calculates Euclidean distance and angle from 2D pose to all 2D landmarks."""
        distance_observations = []
        angle_observations = []
        current_x, current_y = current_pose_val[0]
        current_angle = current_pose_val[1]
        for lm in landmarks:
            ### TODO ###
            ...
            lm_x, lm_y = lm
            dist = math.sqrt((current_x - lm_x) ** 2 + (current_y - lm_y) ** 2)
            rel_x = lm_x - current_x
            rel_y = lm_y - current_y
            theta = math.atan2(rel_y,rel_x)
            angle = theta - current_angle # NOTE idk if i shuld subtract curr angle
            print(f"Observation to landmark {lm}: distance={dist}, angle={angle}")
            ### END TODO ###
            distance_observations.append(dist)
            angle_observations.append(angle)

        return distance_observations, angle_observations

    def get_initial_guess(self) -> Tuple[List[Pose2D], List[Point2D]]:
        """
        Returns an initial guess for the optimizer.
        Here we typically start with a perturbed version of true values or dead-reckoning.
        For this example, we'll start from a guess slightly perturbed from the true poses and
        true landmark locations.
        """
        # Perturb true poses for a better "guess" scenario than just true values
        perturbed_poses = [
            (
                p[0][0] + random.gauss(0, self.initial_guess_std),
                p[0][1] + random.gauss(0, self.initial_guess_std),
                p[1] + random.gauss(0, self.initial_guess_std),
            )
            for p in self.true_poses
        ]

        # Perturb true landmark locations
        perturbed_landmarks_flat = []
        for lm in self.true_landmarks:
            perturbed_landmarks_flat.append(
                (
                    lm[0] + random.gauss(0, self.initial_guess_std),
                    lm[1] + random.gauss(0, self.initial_guess_std),
                )
            )

        return perturbed_poses, perturbed_landmarks_flat