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



# class PIDController:
#     """
#     Proportional-Integral-Derivative (PID) controller with:

#     - output saturation
#     - conditional-integration anti-windup
#     - derivative filtering
#     """

#     def __init__(
#         self,
#         kp,
#         ki,
#         kd,
#         output_min=0.0,
#         output_max=12.0,
#         derivative_filter_tau=0.01,
#     ):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd

#         self.output_min = output_min
#         self.output_max = output_max

#         self.derivative_filter_tau = derivative_filter_tau

#         self.integral = 0.0
#         self.previous_error = 0.0
#         self.derivative = 0.0

#     def reset(self):
#         """Reset controller internal states."""
#         self.integral = 0.0
#         self.previous_error = 0.0
#         self.derivative = 0.0

#     def update(self, error, dt):
#         """
#         Calculate the PID controller output.

#         Parameters
#         ----------
#         error : float
#             Reference minus measured value.

#         dt : float
#             Controller sampling time.

#         Returns
#         -------
#         float
#             Saturated control output.
#         """

#         # Proportional term
#         proportional = self.kp * error

#         # Raw derivative
#         if dt > 0.0:
#             derivative_raw = (
#                 error - self.previous_error
#             ) / dt
#         else:
#             derivative_raw = 0.0

#         # First-order derivative filter
#         alpha = (
#             dt
#             / (self.derivative_filter_tau + dt)
#         )

#         self.derivative += (
#             alpha
#             * (derivative_raw - self.derivative)
#         )

#         # Unsaturated control signal
#         output_unsaturated = (
#             proportional
#             + self.ki * self.integral
#             + self.kd * self.derivative
#         )

#         # Saturation
#         output = max(
#             self.output_min,
#             min(
#                 output_unsaturated,
#                 self.output_max,
#             ),
#         )

#         # Conditional-integration anti-windup
#         if output_unsaturated > self.output_max and error > 0:
#             integral_dot = 0.0

#         elif output_unsaturated < self.output_min and error < 0:
#             integral_dot = 0.0

#         else:
#             integral_dot = error

#         self.integral += integral_dot * dt

#         # Save error for next iteration
#         self.previous_error = error

#         return output

class PIDController:
    """
    PID controller with:

    - output saturation
    - conditional-integration anti-windup
    - filtered derivative
    - derivative on measurement
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        output_min=0.0,
        output_max=12.0,
        derivative_filter_tau=0.01,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_min = output_min
        self.output_max = output_max

        self.derivative_filter_tau = derivative_filter_tau

        self.integral = 0.0
        self.previous_measurement = 0.0
        self.derivative = 0.0

    def reset(self):
        """Reset controller internal states."""
        self.integral = 0.0
        self.previous_measurement = 0.0
        self.derivative = 0.0

    def update(self, error, measurement, dt):
        """
        Calculate PID controller output.

        Parameters
        ----------
        error : float
            Reference minus measured value.

        measurement : float
            Measured process variable.

        dt : float
            Controller sampling time.

        Returns
        -------
        float
            Saturated control output.
        """

        # ----------------------------------------------------
        # Proportional term
        # ----------------------------------------------------

        proportional = self.kp * error

        # ----------------------------------------------------
        # Derivative on measurement
        # ----------------------------------------------------

        if dt > 0.0:
            measurement_rate = (
                measurement - self.previous_measurement
            ) / dt
        else:
            measurement_rate = 0.0

        # First-order derivative filter
        alpha = (
            dt
            / (self.derivative_filter_tau + dt)
        )

        self.derivative += (
            alpha
            * (-measurement_rate - self.derivative)
        )

        derivative = self.kd * self.derivative

        # ----------------------------------------------------
        # Unsaturated controller output
        # ----------------------------------------------------

        output_unsaturated = (
            proportional
            + self.ki * self.integral
            + derivative
        )

        # ----------------------------------------------------
        # Output saturation
        # ----------------------------------------------------

        output = max(
            self.output_min,
            min(
                output_unsaturated,
                self.output_max,
            ),
        )

        # ----------------------------------------------------
        # Conditional-integration anti-windup
        # ----------------------------------------------------

        if output_unsaturated > self.output_max and error > 0:

            integral_dot = 0.0

        elif output_unsaturated < self.output_min and error < 0:

            integral_dot = 0.0

        else:

            integral_dot = error

        self.integral += integral_dot * dt

        # ----------------------------------------------------
        # Store measurement for next iteration
        # ----------------------------------------------------

        self.previous_measurement = measurement

        return output
