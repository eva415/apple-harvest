#!/usr/bin/env python3

"""
FlexToFListener: ROS2 node for fusing flex sensor and ToF distance data to control a UR5 gripper on the apple proxy

- Subscribes to flex sensor (`/flex_sensor_data`) and ToF sensor (`/tof_sensor_data`) topics.
- Uses a Kalman filter to estimate apple position and a PID controller to generate velocity commands.
- Publishes smoothed twist commands to `/servo_node/delta_twist_cmds` and estimated apple position to `/position_apple`.
- Implements a simple state machine (`servo` → `approach` → `pick`) based on position error and distance thresholds.
- Includes acceleration limiting and command smoothing for stable motion.
- Configures and enables MoveIt-Servo via ROS2 service clients.
- Runs a 100 Hz control loop with ReentrantCallbackGroup to handle concurrent callbacks.

Intended for real-time sensor fusion and servo-based manipulation using UR5 + MoveIt-Servo.
"""

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
# Interfaces
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray, Int32, Float32
from geometry_msgs.msg import TwistStamped  # to publish to the UR5
from controller_manager_msgs.srv import SwitchController
import numpy as np
from eva_vacuum_test import PumpIO # my vacuum control file
import time

class FlexToFListener(Node):
    def __init__(self, calibrate=False):
        super().__init__('flex_tof_listener')
        self.cbgroup = ReentrantCallbackGroup()
        self.calibrate = calibrate

        # State machine: start in approach
        self.state = 'approach'
        self.position_threshold = 0.5
        self.tof_servo_threshold = 50
        self.tof_threshold = 46

        # Scale & timing
        self.velocity_scale_factor_xy = 1.0
        self.velocity_scale_factor_z = 4.0
        self.control_period = 0.01  # 100 Hz

        # Sensor placeholders
        self.latest_flex = None
        self.tof_distance = None
        self.latest_pressure = None

        # Publishers & Subscribers
        self.apple_pub = self.create_publisher(Float32MultiArray, '/position_apple', 10)
        self.gripper_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.pressure_pub = self.create_publisher(Float32, '/suction_pressure', 10)
        self.create_subscription(Float32MultiArray, '/flex_sensor_data', self.flex_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Int32, '/tof_sensor_data', self.tof_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Float32, '/vacuum_pressure', self.pressure_callback, 10, callback_group=self.cbgroup)

        # Fixed-rate control loop
        self.prev_time = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(self.control_period, self.control_loop, callback_group=self.cbgroup)

        # Filters & PID state
        self._init_kalman()
        self._init_pid()

        # Command smoothing state
        self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.0
        
        # Initialize Pump
        self.pump = PumpIO(self)
        self.pump.disable_energy_saving()
        self.pump.vacuum_off()

        # Initialize MoveIt-Servo
        self._setup_servo_clients()
        self._enable_servo_mode(frame="tool0")
        self.get_logger().info('FlexToFListener (smoothed) started.')


    def flex_callback(self, msg):
        vals = np.array(msg.data) / 4.0
        self.latest_flex = vals.reshape((4, 1))

    def tof_callback(self, msg):
        self.tof_distance = msg.data

    def pressure_callback(self, msg):
        self.latest_pressure = msg.data  # store suction pressure


    def control_loop(self):
        print(f"STATE: {self.state}, tof: {self.tof_distance}, pressure: {self.latest_pressure}")
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self.prev_time
        self.prev_time = now
        
        if self.latest_flex is None or self.tof_distance is None:
            return

        # Kalman + PID
        self._kalman_update(self.latest_flex)
        vx, vy = self._pid_compute(self.x, dt)
        vz = 0.0
        ex = abs(self.smoothed_x - self.current_x)
        ey = abs(self.smoothed_y - self.current_y)

        # --- State transitions ---
        if self.state == 'servo':
            # Switch to 'approach' if centered OR below servo threshold
            if (ex < self.position_threshold and ey < self.position_threshold) or self.tof_distance <= self.tof_servo_threshold:
                self.state = 'approach'
        elif self.state == 'approach':
            # Only go back to servo if above threshold and off-center
            if self.tof_distance > self.tof_servo_threshold and (ex > self.position_threshold or ey > self.position_threshold):
                self.state = 'servo'
            elif self.tof_distance <= self.tof_threshold:
                self.state = 'pick'
                self.pick_start_time = now
                self.latest_pressure = None
                self.get_logger().info(f'Picking: turning on vacuum (tof = {self.tof_distance})')
                self.pump.vacuum_on()

        elif self.state == 'pick':
            elapsed = now - self.pick_start_time
            pressure = self.latest_pressure if self.latest_pressure is not None else float('inf')
            print(f"[DEBUG] pick elapsed={elapsed:.2f}, pressure={pressure}")
            # Success
            if pressure <= -50:
                self.get_logger().info(f'Vacuum succeeded (pressure={pressure})')
                self.state = 'release'
                self.release_start_time = now
            # Timeout
            elif elapsed > 10.0:
                self.get_logger().warn(f'Pick failed: timeout (pressure={pressure})')
                self.pump.vacuum_off()
                self.state = 'failed'  # use a failure state instead of immediate shutdown

        
        elif self.state == 'release':
            elapsed_release = now - self.release_start_time
            print(f"RELEASE: {elapsed_release}")
            if elapsed_release < 4.0:
                self.get_logger().info(f"retreating...")
            elif elapsed_release > 4.0 and elapsed_release < 8.0:
                self.get_logger().info(f"releasing apple now...")
                self.pump.vacuum_off()
            elif elapsed_release >= 8.0:
                print('done')
                self.state = 'done'

        cmd_wz = 0.0   # default: no rotation
        # --- Command selection ---
        if self.state == 'servo':
            cmd_vx, cmd_vy, cmd_vz = -vx, -vy, 0.0
        elif self.state == 'approach':
            cmd_vx, cmd_vy = 0.0, 0.0
            cmd_vz = 0.1 * self.velocity_scale_factor_z
        elif self.state == 'release':
            elapsed_release = now - self.release_start_time
            if elapsed_release < 4.0:
                cmd_vx, cmd_vy, cmd_vz = 0.0, 0.0, -2.0
                cmd_wz = -4.0
            elif elapsed_release < 8.0:
                cmd_vx, cmd_vy, cmd_vz = 0.0, 0.0, 0.0
                cmd_wz = 4.0
        else: # pick state or done state
            cmd_vx = cmd_vy = cmd_vz = 0.0

        # --- Acceleration limit + smoothing ---
        dvx = np.clip(cmd_vx - self.prev_cmd_x, -self.acc_max * dt, self.acc_max * dt)
        dvy = np.clip(cmd_vy - self.prev_cmd_y, -self.acc_max * dt, self.acc_max * dt)
        dvz = np.clip(cmd_vz - self.prev_cmd_z, -self.acc_max * dt, self.acc_max * dt)

        raw_x = self.prev_cmd_x + dvx
        raw_y = self.prev_cmd_y + dvy
        raw_z = self.prev_cmd_z + dvz

        out_x = self.alpha_cmd * raw_x + (1 - self.alpha_cmd) * self.prev_cmd_x
        out_y = self.alpha_cmd * raw_y + (1 - self.alpha_cmd) * self.prev_cmd_y
        out_z = self.alpha_cmd * raw_z + (1 - self.alpha_cmd) * self.prev_cmd_z
        self.prev_cmd_x, self.prev_cmd_y, self.prev_cmd_z = out_x, out_y, out_z

        # Publish twist
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'tool0'
        cmd.twist.linear.x = out_x
        cmd.twist.linear.y = out_y
        cmd.twist.linear.z = out_z
        cmd.twist.angular.x = cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = cmd_wz
        self.gripper_pub.publish(cmd)

        # Debug apple pos
        apple = Float32MultiArray(data=[float(self.x[1]), float(self.x[0])])
        self.apple_pub.publish(apple)

    def _init_kalman(self):
        n, m = 2, 4
        self.z = np.zeros((m,1))
        self.x = np.zeros((n,1))
        self.P = np.eye(n)
        self.A = np.eye(n)
        self.H = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        self.Q = np.eye(n)*0.05
        self.R = np.eye(m)*0.05

    def _init_pid(self):
        self.current_x = self.current_y = 0.0
        self.current_x_vel = self.current_y_vel = 0.0
        self.smoothed_x = self.smoothed_y = 0.0
        self.alpha_pos = 0.5
        self.alpha_cmd = 0.3
        self.K_p = 0.3
        self.K_i = 0.0
        self.K_d = 0.01
        self.integral_x = self.integral_y = 0.0
        self.prev_err_x = self.prev_err_y = 0.0
        self.vel_max = 0.1
        self.acc_max = 0.5

    def _setup_servo_clients(self):
        mcb = MutuallyExclusiveCallbackGroup()
        self.switch_cli = self.create_client(SwitchController, "/controller_manager/switch_controller", callback_group=mcb)
        self.start_cli = self.create_client(Trigger, "/servo_node/start_servo", callback_group=mcb)
        self.param_cli = self.create_client(SetParameters, "/servo_node/set_parameters", callback_group=mcb)
        while not self.switch_cli.wait_for_service(1.0):
            self.get_logger().info("Waiting switch_controller...")
        while not self.start_cli.wait_for_service(1.0):
            self.get_logger().info("Waiting start_servo...")
        while not self.param_cli.wait_for_service(1.0):
            self.get_logger().info("Waiting set_parameters...")

    def _kalman_update(self, z):
        x_p = self.A @ self.x
        P_p = self.A @ self.P @ self.A.T + self.Q
        K = P_p @ self.H.T @ np.linalg.inv(self.H @ P_p @ self.H.T + self.R)
        self.x = x_p + K @ (z - self.H @ x_p)
        self.P = P_p - K @ self.H @ P_p

    def _pid_compute(self, x_est, dt):
        self.smoothed_x = self.alpha_pos * x_est[1, 0] + (1 - self.alpha_pos) * self.smoothed_x
        self.smoothed_y = self.alpha_pos * x_est[0, 0] + (1 - self.alpha_pos) * self.smoothed_y

        err_x = self.smoothed_x - self.current_x
        err_y = self.smoothed_y - self.current_y
        self.integral_x += err_x * dt
        self.integral_y += err_y * dt
        der_x = (err_x - self.prev_err_x) / dt
        der_y = (err_y - self.prev_err_y) / dt

        vx = self.K_p * err_x + self.K_i * self.integral_x + self.K_d * der_x
        vy = self.K_p * err_y + self.K_i * self.integral_y + self.K_d * der_y

        vx = np.clip(vx, -self.vel_max, self.vel_max)
        vy = np.clip(vy, -self.vel_max, self.vel_max)

        self.current_x += vx * dt
        self.current_y += vy * dt
        self.prev_err_x, self.prev_err_y = err_x, err_y
        self.current_x_vel, self.current_y_vel = vx, vy
        return vx * self.velocity_scale_factor_xy, vy * self.velocity_scale_factor_xy

    def _enable_servo_mode(self, frame: str = "tool0"):
        req = SwitchController.Request()
        req.activate_controllers = ["forward_position_controller"]
        req.deactivate_controllers = ["joint_trajectory_controller"]
        req.strictness = SwitchController.Request.STRICT
        req.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        fut = self.switch_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)

        start_req = Trigger.Request()
        fut2 = self.start_cli.call_async(start_req)
        rclpy.spin_until_future_complete(self, fut2)

        prm = SetParameters.Request()
        val = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=frame)
        prm.parameters = [Parameter(name='moveit_servo.robot_link_command_frame', value=val)]
        fut3 = self.param_cli.call_async(prm)
        rclpy.spin_until_future_complete(self, fut3)


def main():
    rclpy.init()
    node = FlexToFListener()
    exe = MultiThreadedExecutor()
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: shutting down...")
    finally:
        # Ensure vacuum is turned off safely
        if hasattr(node, "pump"):
            node.get_logger().info("Turning off vacuum before exit...")
            node.pump.vacuum_off()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
