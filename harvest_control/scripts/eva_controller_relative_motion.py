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
from std_msgs.msg import Float32MultiArray, Int32, Float32, String
from geometry_msgs.msg import TwistStamped  # to publish to the UR5
from controller_manager_msgs.srv import SwitchController
import numpy as np
from eva_vacuum_test import PumpIO # my vacuum control file
import time
from collections import deque
import atexit
import sys


class FlexToFListener(Node):
    def __init__(self, calibrate=False):
        super().__init__('flex_tof_listener')
        self.cbgroup = ReentrantCallbackGroup()
        self.calibrate = calibrate
        self.active = False  # control loop paused until trigger

        # State machine: start in idle
        self.state = 'idle'
        self.position_threshold = 0.5
        self.tof_servo_threshold = self.tof_relative_motion_threshold = 58
        self.timeout = 45.0

        # Scale & timing
        self.velocity_scale_factor_xy = 0.5
        self.velocity_scale_factor_z = 2.0
        self.control_period = 0.01  # 100 Hz

        # Sensor placeholders
        self.latest_flex = None
        self.tof_distance = None
        self.latest_pressure = None

        # Publishers & Subscribers
        self.apple_pub = self.create_publisher(Float32MultiArray, '/position_apple', 10)
        self.gripper_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.pressure_pub = self.create_publisher(Float32, '/suction_pressure', 10)
        self.relative_motion_pub = self.create_publisher(String, '/relative_motion_status', 10)
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

        # ToF history buffer (stores last 10 readings)
        self.tof_history = deque(maxlen=10)
        self.controller = 'default'

        # WIGGLE sequence
        self.wiggle_sequence = [
            {'vx': 0.0, 'vy': 0.0, 'vz': 0.25, 'wx': 0.55, 'wy': 0.0},   # +z +x
            {'vx': 0.0, 'vy': 0.0, 'vz': -0.25, 'wx': -0.55, 'wy': 0.0}, # return
            {'vx': 0.0, 'vy': 0.0, 'vz': 0.25, 'wx': -0.55, 'wy': 0.0},  # +z -x
            {'vx': 0.0, 'vy': 0.0, 'vz': -0.25, 'wx': 0.55, 'wy': 0.0},  # return
            {'vx': 0.0, 'vy': 0.0, 'vz': 0.25, 'wx': 0.0, 'wy': 0.55},   # +z +y
            {'vx': 0.0, 'vy': 0.0, 'vz': -0.25, 'wx': 0.0, 'wy': -0.55}, # return
            {'vx': 0.0, 'vy': 0.0, 'vz': 0.25, 'wx': 0.0, 'wy': -0.55},  # +z -y
            {'vx': 0.0, 'vy': 0.0, 'vz': -0.25, 'wx': 0.0, 'wy': 0.55},  # return
        ]
        self.wiggle_step = 0
        self.wiggle_step_duration = 0.7  # seconds per motion
        self.wiggle_step_start_time = None

        # --- alignment tuning ---
        # fraction of approach rate below which we consider 'almost stopped'
        # e.g. 0.2 means measured rate must be < 20% of commanded approach rate
        self.alignment_beta = 0.4

        # require X consecutive checks of 'almost stopped' before declaring alignment
        self.alignment_debounce_count = 5

        # running counter for debounce
        self.alignment_counter = 0

        # require ToF to be at least this close before picking.
        # TUNE this to be slightly smaller than your tof_relative_motion_threshold so pick happens later.
        self.pick_distance_threshold = self.tof_relative_motion_threshold  # tweak as needed

        # Initialize MoveIt-Servo
        self._setup_servo_clients()
        # self._enable_servo_mode(frame="tool0")
        self.get_logger().info('FlexToFListener (smoothed) started.')

    def flex_callback(self, msg):
        vals = np.array(msg.data) / 4.0
        self.latest_flex = vals.reshape((4, 1))

    def tof_callback(self, msg):
        self.tof_distance = msg.data
        self.tof_history.append(self.tof_distance)
        self.get_logger().debug(f"ToF history: {list(self.tof_history)}")

    def get_tof_diff(self):
        # Need at least 10 readings to compute a stable diff
        if len(self.tof_history) < 10:
            return None
        
        # Take the first 5 readings and last 5 readings
        first_avg = sum(list(self.tof_history)[:5]) / 5
        last_avg = sum(list(self.tof_history)[-5:]) / 5
        
        return last_avg - first_avg

    def pressure_callback(self, msg):
        self.latest_pressure = msg.data  # store suction pressure


    def control_loop(self):
        status_msg = String()
        status_msg.data = f"{self.state}"  # e.g., 'servo', 'approach', 'wiggle', 'pick', etc.
        self.relative_motion_pub.publish(status_msg)

        if not self.active:
            return  # don't run until triggered
        
        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed_total = now - getattr(self, "motion_start_time", now)
        # --- Global timeout guard ---
        if elapsed_total > self.timeout and self.state != "pick":
            self.get_logger().warn("Not picking. Relative motion timed out (global). Forcing FAIL.")
            self.state = "FAIL"

        # --- End condition (shared for FAIL or done) ---
        if self.state in ("FAIL", "done"):
            self.get_logger().info(f"State reached terminal condition: {self.state}. Cleaning up.")
            self.active = False
            time.sleep(1)
            self.state = "done"
            self.pump.vacuum_off()
            self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.0
            self.wiggle_step_start_time = None
            self.pick_start_time = None
            self.release_start_time = None
            self.latest_flex = None
            self.tof_distance = None
            self.latest_pressure = None
            self.tof_history = deque(maxlen=10)
            self.controller = 'default'
            return

        diff = self.get_tof_diff()

        # testing my tof difference function to make a new controller
        if diff is not None:
            # self.get_logger().info(f"ToF rate: {diff}")
            pass

        self.get_logger().info(f"CONTROLLER: {self.controller}, STATE: {self.state}, tof: {self.tof_distance}, pressure: {self.latest_pressure}")
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
        if self.controller == 'default':
            if self.state == 'servo':
                # Switch to 'approach' if centered OR below servo threshold
                if (ex < self.position_threshold and ey < self.position_threshold) or self.tof_distance <= self.tof_servo_threshold:
                    self.state = 'approach'
            elif self.state == 'approach':
                if self.tof_distance > self.tof_servo_threshold and (ex > self.position_threshold or ey > self.position_threshold):
                    self.state = 'servo'
                elif self.tof_distance <= self.tof_relative_motion_threshold:
                    self.controller = 'relative_controller'
        if self.controller == 'relative_controller':
            if self.state == 'servo':
                # Switch to 'approach' if centered OR below servo threshold
                if (ex < self.position_threshold and ey < self.position_threshold) or self.tof_distance <= self.tof_servo_threshold:
                    self.state = 'approach'
            if self.state == 'approach':
                if ex > self.position_threshold or ey > self.position_threshold:
                    self.state = 'servo'
                if self.get_tof_diff() < 0: # apple is getting closer
                    pass# self.get_logger().info("apple is getting closer")
                elif self.get_tof_diff() > 0: # apple is ready to pick
                    # self.get_logger().info("apple is nicely aligned")
                    self.state = 'wiggle'
                    self.pick_start_time = now
                    self.latest_pressure = None
                    self.get_logger().info(f'Wiggling: turning on vacuum (tof = {self.tof_distance})')
                    self.pump.vacuum_on()
            if self.state == 'reverse':
                if self.get_tof_diff() < 0: # apple is getting closer
                    pass # self.get_logger().info("apple is getting closer")
                elif self.get_tof_diff() > 0: # apple is being pushed away
                    # self.get_logger().info("apple is nicely aligned")
                    self.state = 'wiggle'
                    self.pick_start_time = now
                    self.latest_pressure = None
                    self.get_logger().info(f'Wiggling: turning on vacuum (tof = {self.tof_distance})')
                    self.pump.vacuum_on()
            elif self.state == 'wiggle':
                # immediate brake:
                self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.00
                elapsed = now - self.pick_start_time
                pressure = self.latest_pressure if self.latest_pressure is not None else float('inf')
                print(f"[DEBUG] pick elapsed={elapsed:.2f}, pressure={pressure}")
                # Success
                if pressure <= -56:
                    self.get_logger().info(f'Vacuum succeeded (pressure={pressure})')
                    self.state = 'pick'
                    self.release_start_time = now
                # Timeout
                elif elapsed > self.timeout:
                    self.get_logger().warn(f'Pick failed: timeout (pressure={pressure})')
                    self.pump.vacuum_off()
                    self.state = 'FAIL'  # use a failure state instead of immediate shutdown
            elif self.state == 'pick':
                elapsed_release = now - self.release_start_time
                print(f"PICK MOTION: {elapsed_release}")
                if elapsed_release < 3.0:
                    self.get_logger().info(f"retreating...")
                elif elapsed_release > 3.0 and elapsed_release < 6.0:
                    self.get_logger().info(f"releasing apple now...")
                    self.pump.vacuum_off()
                    self.state = 'done'


        cmd_wz = cmd_wy = cmd_wx = 0.0   # default: no rotation
        cmd_vx = cmd_vy = cmd_vz = 0.0 # default: no translation
        # --- Command selection ---
        if self.state == 'servo':
            cmd_vx, cmd_vy, cmd_vz = -vx, -vy, 0.2
        elif self.state == 'approach':
            cmd_vx, cmd_vy = 0.0, 0.0
            cmd_vz = 0.15 * self.velocity_scale_factor_z
        elif self.state == 'reverse':
            cmd_vx, cmd_vy = 0.0, 0.0
            cmd_vz = -0.15 * self.velocity_scale_factor_z
        elif self.state == 'pick':
            elapsed_release = now - self.release_start_time
            if elapsed_release < 3.0:
                cmd_vx, cmd_vy, cmd_vz = 0.0, 0.0, -8.0
                cmd_wz = -4.0
            else:
                self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.00
                cmd_vx, cmd_vy, cmd_vz = 0.0, 0.0, 0.0
        elif self.state == 'wiggle':
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.wiggle_step_start_time is None:
                self.wiggle_step_start_time = now

            # Check if current step duration elapsed
            if now - self.wiggle_step_start_time > self.wiggle_step_duration:
                self.wiggle_step = (self.wiggle_step + 1) % len(self.wiggle_sequence)
                self.wiggle_step_start_time = now

            # Apply current step
            step = self.wiggle_sequence[self.wiggle_step]
            cmd_vx = step['vx']
            cmd_vy = step['vy']
            cmd_vz = step['vz']
            cmd_wx = step['wx']
            cmd_wy = step['wy']
            # cmd_vx = 0.0
            # cmd_vy = 0.0
            # cmd_vz = 0.0
            # cmd_wx = 0.0
            # cmd_wy = 0.0

            # Override previous commands for smoothing as before
            self.prev_cmd_x = cmd_vx
            self.prev_cmd_y = cmd_vy
            self.prev_cmd_z = cmd_vz
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
        cmd.twist.angular.x = cmd_wx
        cmd.twist.angular.y = cmd_wy
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
        self.Q = np.eye(n)*0.04
        self.R = np.eye(m)*0.2

    def _init_pid(self):
        self.current_x = self.current_y = 0.0
        self.current_x_vel = self.current_y_vel = 0.0
        self.smoothed_x = self.smoothed_y = 0.0
        self.alpha_pos = 0.9
        self.alpha_cmd = 0.3
        self.K_p = 0.45
        self.K_i = 0.0
        self.K_d = 0.01
        self.integral_x = self.integral_y = 0.0
        self.prev_err_x = self.prev_err_y = 0.0
        self.vel_max = 0.2 # good line to change if x-y motion is lagging
        self.acc_max = 8.0 # good line to change if x-y motion is lagging

    def _setup_servo_clients(self):
        mcb = MutuallyExclusiveCallbackGroup()
        # self.switch_cli = self.create_client(SwitchController, "/controller_manager/switch_controller", callback_group=mcb)
        self.start_cli = self.create_client(Trigger, "/servo_node/start_servo", callback_group=mcb)
        self.param_cli = self.create_client(SetParameters, "/servo_node/set_parameters", callback_group=mcb)
        self.relative_motion_srv = self.create_service(Trigger, 'relative_z_motion', self.relative_motion_cb)
        # while not self.switch_cli.wait_for_service(1.0):
        #     self.get_logger().info("Waiting switch_controller...")
        while not self.start_cli.wait_for_service(1.0):
            self.get_logger().info("Waiting start_servo...")
        while not self.param_cli.wait_for_service(1.0):
            self.get_logger().info("Waiting set_parameters...")

    def relative_motion_cb(self, request, response):
        if self.active:
            response.success = False
            response.message = "Already running a motion."
            self.get_logger().warn(response.message)
            return response

        # Reset ToF history and relevant timers
        self.tof_history.clear()  # clear stale distance readings
        self.state = "approach"
        self.controller = "default"
        self.active = True
        self.motion_start_time = self.get_clock().now().nanoseconds * 1e-9

        response.success = True
        response.message = "Relative motion started"
        self.get_logger().info(response.message)
        return response

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
        req.timeout = rclpy.duration.Duration(seconds=self.timeout).to_msg()
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
    
    def _shutdown_sequence(self):
        print("Running shutdown sequence...")

        # 1. shut off vacuum
        if hasattr(self, "pump"):
            print("Turning off vacuum...")
            self.pump.vacuum_off()
            time.sleep(0.5)

        # 2. destroy node
        print("Destroying node...")
        self.destroy_node()
            

def main():
    rclpy.init()
    node = FlexToFListener()
    exe = MultiThreadedExecutor()
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, exiting...")
    finally:
        node._shutdown_sequence()  # hardware cleanup
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()
