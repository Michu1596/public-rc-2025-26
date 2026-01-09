import numpy as np
from slam_core.backend_2d import SLAM2DBackend
from slam_core.frontend_2d import SLAM2DFrontend
from slam_core.visualization import plot_results_2d


def main():
    print("Initializing SLAM Simulation...")

    # 1. Setup Frontend (Simulation)
    # ------------------------------
    # Define noise levels
    # For debugging, set this to zero
    odometry_noise = 0.05
    sensor_noise = 0.1
    initial_guess_noise = 0.1

    frontend = SLAM2DFrontend(
        initial_pose=((0, 0), 0),
        trajectory_shape="square",
        num_steps=4,
        odometry_noise_std=odometry_noise,
        sensor_noise_std=sensor_noise,
        initial_guess_std=initial_guess_noise,
    )

    # Add landmarks
    frontend.add_landmark(1.0, 1.0)
    frontend.add_landmark(2.0, 2.0)

    # Generate data
    frontend.generate_ground_truth()
    frontend.generate_measurements()

    print(f"True Poses: {frontend.true_poses}")
    print(f"True Landmarks: {frontend.true_landmarks}")
    print(f"Noisy Odometry Measurements: ")
    for odom in frontend.odometry_measurements:
        print(odom)
    print(
        f"Noisy Sensor Measurements: {frontend.sensor_measurements}"
    )  # Can be very long
    print(f"Odometry Noise Std: {odometry_noise}")
    print(f"Sensor Noise Std: {sensor_noise}")

    # 2. Setup Backend (Optimization)
    # -------------------------------
    backend = SLAM2DBackend()

    # Get Initial Guess (now perturbed by default in frontend)
    initial_poses_guess, initial_landmarks_guess = frontend.get_initial_guess()

    print("\nStarting Optimization...")
    result = backend.optimize(
        initial_poses_guess,
        initial_landmarks_guess,
        frontend.odometry_measurements,
        frontend.sensor_measurements,
        odometry_noise_std=odometry_noise,
        sensor_noise_std=sensor_noise,
        ground_truth_initial_pose=frontend.initial_pose,
    )

    # 3. Results
    # ----------
    print("\nOptimization Complete!")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final Cost (Minimized Likelihood Value): {result.fun}")

    # Extract results
    num_poses = len(initial_poses_guess) * 3
    optimized_poses = result.x[:num_poses]
    optimized_landmarks_flat = result.x[num_poses:]
    optimized_landmarks = []
    for i in range(0, len(optimized_landmarks_flat), 2):
        optimized_landmarks.append(
            (optimized_landmarks_flat[i], optimized_landmarks_flat[i + 1])
        )

    print("\n--- Final Estimates ---")
    print(f"Initial Poses Guess: {initial_poses_guess}")
    print(f"Initial Landmarks Guess (flat): {initial_landmarks_guess}")
    print(f"Estimated Poses: {optimized_poses}")
    print(f"Estimated Landmarks: {optimized_landmarks}")

    # Validation
    print("\n--- Errors ---")
    print(optimized_poses, frontend.true_poses)
    f_poses = [[p[0][0], p[0][1], p[1]] for p in frontend.true_poses]
    pose_error = np.linalg.norm(np.array(optimized_poses) - np.array(f_poses).flatten())
    landmark_error = 0.0
    for i in range(len(frontend.true_landmarks)):
        landmark_error += np.linalg.norm(
            np.array(optimized_landmarks[i]) - np.array(frontend.true_landmarks[i])
        )

    print(f"Pose Error (L2 norm vs true poses): {pose_error:.4f}")
    print(f"Landmark Error (Sum of L2 norms vs true landmarks): {landmark_error:.4f}")

    # 4. Visualization
    # ----------------
    print("\nGenerating Visualization...")
    plot_results_2d(
        true_poses=frontend.true_poses,
        true_landmarks=frontend.true_landmarks,
        odometry_measurements=frontend.odometry_measurements,
        optimized_poses=optimized_poses,
        optimized_landmarks=optimized_landmarks,
        initial_pose=frontend.initial_pose,
    )


if __name__ == "__main__":
    main()
