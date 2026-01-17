import numpy as np

class KalmanFilter:
    def __init__(self, process_var, measurement_var):
        # dt: time interval
        # process_var: process variance, represents uncertainty in the model
        # measurement_var: measurement variance, represents measurement noise

        # Measurement Matrix
        ## TODO ##
        # Set the measurement matrix H
        self.H = np.eye(3,9)

        # Process Covariance Matrix
        self.Q = np.eye(9) * process_var

        # Measurement Covariance Matrix
        self.R = np.eye(3) * measurement_var

        # Initial State Covariance Matrix
        self.P = np.eye(9)

        # Initial State
        self.x = np.zeros((9, 1)) # [x,y,z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc] or pitch, roll, yaw

    def predict(self, dt):
        ### TODO ###
        # State Transition Matrix
        A = np.array([[1,0,0,dt,0,0,0.5*dt*dt,0,0],
                      [0,1,0,0,dt,0,0,0.5*dt*dt,0],
                      [0,0,1,0,0,dt,0,0,0.5*dt*dt],
                      [0,0,0,1,0,0,dt,0,0],
                      [0,0,0,0,1,0,0,dt,0],
                      [0,0,0,0,0,1,0,0,dt],
                      [0,0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,0,0,1]])
        # print(f"x pre predict: {self.x}")
        self.x = A @ self.x
        print(f"x {self.x} dt : {dt}")
        assert self.x.shape == (9,1), "bad shape"
        # print(f"x po predict: {self.x}")
        self.P = A @ self.P @ A.T + self.Q
        ###

    def update(self, measurement : np.array):
        # Update the state with the new measurement
        ### TODO ###
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        assert self.x.shape == (9,1), "bad shape"
        self.P = (np.eye(9) - K @ self.H) @ self.P
        ### ###
