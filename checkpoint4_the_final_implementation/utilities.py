import numpy as np

class PIDController:
    """PID controller for throttle (velocity tracking)."""

    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        '''
        write the update function on your own
        input is error and dt
        output should be the thrust that is provided to drone
        '''
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        thrust = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return thrust





def pure_pursuit_steering(x, y, yaw, v, waypoints, L=2.5, k_dd=0.5, ld_min=2.0, max_steer=np.radians(30)):
    """
    Pure Pursuit steering controller.

    Args:
        x, y       : rear axle position of the car
        yaw        : vehicle heading angle (in radians)
        v          : vehicle speed
        waypoints  : Nx2 array of path waypoints
        L          : wheelbase (distance between front and rear axles)
        k_dd       : lookahead gain (scales with speed)
        ld_min     : minimum lookahead distance
        max_steer  : steering angle limits (in radians)

    Returns:
        steer      : steering angle in radians
        target_idx : index of the goal waypoint
    """

    # Step 1: Compute the lookahead distance based on speed
    ld = k_dd * v + ld_min


    # Step 2: Find the nearest waypoint to the rear axle
    #   Compute distances from (x, y) to every waypoint, pick the closest index.
    min_dist = float('inf')
    nearest_idx = 0
    for i, (wx, wy) in enumerate(waypoints):
        dist = np.hypot(wx - x, wy - y)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i


    # Step 3: Search forward from the nearest waypoint to find the goal point
    #   Starting from the nearest index, walk forward along the waypoints until
    #   you find the first waypoint whose distance from (x, y) >= ld.
    #   If none is found, use the last waypoint in the array.
    target_idx = nearest_idx
    while target_idx < len(waypoints):
        wx, wy = waypoints[target_idx]
        dist = np.hypot(wx - x, wy - y)
        if dist >= ld:
            break
        target_idx += 1
        if target_idx >= len(waypoints):
            target_idx = len(waypoints) - 1
            break

    # Step 4: Compute alpha — the angle between the vehicle heading and the
    #   direction from the rear axle to the goal point.
    alpha = np.arctan2(waypoints[target_idx][1] - y, waypoints[target_idx][0] - x) - yaw


    # Step 5: Compute the steering angle using the Pure Pursuit formula:
    #   steer = arctan(2 * L * sin(alpha) / ld)
    steer = np.arctan(2 * L * np.sin(alpha) / ld)

    # Step 6: Clip steering to max_steer
    steer = np.clip(steer, -max_steer, max_steer)

    return steer, target_idx
