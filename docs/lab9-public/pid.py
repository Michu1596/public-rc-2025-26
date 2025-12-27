from numpy import clip


class PID:
    def __init__(
        self,
        gain_prop: float,
        gain_int: float,
        gain_der: float,
        sensor_period: float,
        output_limits: tuple[float, float],
    ):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: define additional attributes you might need
        self.integral = 0
        self.output_limits = output_limits
        # END OF TODO

    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(
        self, commanded_variable: float, sensor_readings: list[float]
    ) -> float:
        current, previous = sensor_readings[0], sensor_readings[1]

        proportional = self.gain_prop * (commanded_variable - sensor_readings[0])

        # use trapezoidal rule to compute integral
        self.integral += (
            0.5
            * (
                commanded_variable
                - sensor_readings[0]
                + commanded_variable
                - sensor_readings[1]
            )
            * self.sensor_period
        )
        # clip integral to avoid windup
        max_val, min_val = (
            abs(self.output_limits[0] / (self.gain_int + 1e-3)),
            -abs(self.output_limits[1] / (self.gain_int + 1e-3)),
        ) # we devide by gain_int to avoid excessive clipping when gain_int is high
        self.integral = clip(self.integral, min_val, max_val)
        integral = self.gain_int * self.integral

        derivative = self.gain_der * (current - previous) / self.sensor_period

        return clip(
            proportional + integral - derivative,
            self.output_limits[0],
            self.output_limits[1],
        )

    # END OF TODO
