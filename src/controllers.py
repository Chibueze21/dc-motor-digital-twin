class PIController:
    """
    Proportional-Integral (PI) controller with output saturation
    and conditional-integration anti-windup.
    """

    def __init__(
        self,
        kp,
        ki,
        output_min=0.0,
        output_max=12.0,
    ):
        self.kp = kp
        self.ki = ki

        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0

    def reset(self):
        """Reset the integral state."""
        self.integral = 0.0

    def update(self, error, dt):
        """
        Calculate controller output.

        Parameters
        ----------
        error : float
            Reference minus measured value.
        dt : float
            Simulation time step.

        Returns
        -------
        float
            Saturated controller output.
        """

        # Unsaturated controller output
        output_unsaturated = (
            self.kp * error
            + self.ki * self.integral
        )

        # Apply actuator saturation
        output = max(
            self.output_min,
            min(
                output_unsaturated,
                self.output_max,
            ),
        )

        # Conditional integration anti-windup
        if output_unsaturated > self.output_max and error > 0:
            integral_dot = 0.0

        elif output_unsaturated < self.output_min and error < 0:
            integral_dot = 0.0

        else:
            integral_dot = error

        # Integrate the error
        self.integral += integral_dot * dt

        return output
