#!/usr/bin/env python3
"""
FlexToFListener (edited to support start/stop services)
"""

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
# Interfaces
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger, Empty
from std_msgs.msg import Float32MultiArray, Int32, Float32
from geometry_msgs.msg import TwistStamped, WrenchStamped  # to publish to the UR5
from controller_manager_msgs.srv import SwitchController
import numpy as np
from eva_vacuum_test import PumpIO # my vacuum control file
import time
from collections import deque
import joblib
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from collections import deque
# at top of file (merge with existing imports)
from rclpy.qos import QoSProfile
from message_filters import ApproximateTimeSynchronizer, Subscriber


FLEX_CONTROLLER = False  # change to False to stop flex sensor servoing

TOF_CONTROLLER = False   # change to False to use a distance-only trigger (not relative distance)

PRESSURE_CONTROLLER = False  # change to False to stop pressure threshold logic
PRESSURE_THRESHOLD = -56    # this is a "good enough" pressure to reach, to continue onto picking motion
PRESSURE_CONTROLLER_TIMEOUT = 5.0   # waits 5 seconds before starting picking motion

RF_CONTROLLER = True
PICKING_TIME = 2.0
RELEASE_TIME = 4.0

class FlexToFListener(Node):
    def __init__(self, calibrate=False):
        super().__init__('flex_tof_listener')
        self.cbgroup = ReentrantCallbackGroup()
        self.calibrate = calibrate

        # --- RUNNING FLAG (start/stop) ---
        self.running = False  # <-- ADDED: gate control loop

        # State machine: start in approach
        self.state = 'approach'
        self.position_threshold = 0.5
        self.tof_servo_threshold = 45
        self.tof_relative_motion_threshold = 45

        # Scale & timing
        self.velocity_scale_factor_xy = 1.0
        self.velocity_scale_factor_z = 3.0
        self.control_period = 0.01  # 100 Hz

        # Sensor placeholders
        self.latest_flex = None
        self.latest_tof = None
        self.latest_pressure = None
        self.latest_force = None

        self._auto_stop_timer = None

        # Publishers & Subscribers
        self.apple_pub = self.create_publisher(Float32MultiArray, '/position_apple', 10)
        self.gripper_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.pressure_pub = self.create_publisher(Float32, '/suction_pressure', 10)
        self.create_subscription(Float32MultiArray, '/flex_sensor_data', self.flex_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Int32, '/tof_sensor_data', self.tof_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Float32, '/vacuum_pressure', self.pressure_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.force_callback, 10, callback_group=self.cbgroup)

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

        # ToF history buffer (stores last 15 readings)
        self.tof_history = deque(maxlen=15)
        self.controller = 'default'

        # --- RUNNING FLAG (start/stop) ---
        self.running = False  # gate control loop

        # --- CLIENTS READY FLAG (we'll set up blocking clients lazily on start) ---
        self._clients_ready = False

        # --- start/stop services
        self.start_service = self.create_service(Empty, 'relative_motion/start_controller', self.handle_start)
        self.stop_service  = self.create_service(Empty, 'relative_motion/stop_controller', self.handle_stop)

        self.get_logger().info('FlexToFListener initialized (start/stop services created).')

        # --- RF MODEL SETUP
        self.rf_model = None
        self.rf_model_loaded = False
        
        # Default safe values (in case load fails)
        self.rf_window_size = 5 
        self.rf_feature_order = ["Flex", "Pressure", "Force", "TOF"]
        self.rf_scaler = None

        self._force_raw_buffer = deque(maxlen=21) # Pre-filter buffer

        # RF inference timing
        self.last_rf_time = 0.0
        self.rf_period = 0.009

        self._last_built_X = None # last built aligned window (set by rf_sync_cb)

        # init sliding-window history dict
        self._rf_hist = {k: deque(maxlen=self.rf_window_size) for k in self.rf_feature_order}

        if RF_CONTROLLER:
            model_path = Path(
                get_package_share_directory("harvest_control")
            ) / "resource" / "rf_pick_classifier.joblib"

            try:
                data = joblib.load(model_path)

                # 1. Extract Model
                if isinstance(data, dict) and "model" in data:
                    self.rf_model = data["model"]
                elif hasattr(data, "predict"):
                    self.rf_model = data
                else:
                    raise RuntimeError(f"Unrecognized model format: {type(data)}")

                # 2. Extract Metadata (Overwrite defaults)
                self.rf_window_size = int(data.get("window_size", 5))
                self.rf_feature_order = list(data.get("feature_order", ["Flex", "Pressure", "Force", "TOF"]))
                self.rf_scaler = data.get("scalar", None)

                # 3. Initialize History Buffers matching the trained feature order
                self._rf_hist = {k: deque(maxlen=self.rf_window_size) for k in self.rf_feature_order}
                self._force_raw_buffer = deque(maxlen=21)
                
                self.rf_model_loaded = True

                self.get_logger().info(
                    f"RF loaded. Window: {self.rf_window_size}, Features: {self.rf_feature_order}"
                )

            except Exception as e:
                self.get_logger().error(f"Failed to load RF model. RF Control DISABLED. Error: {e}")
                self.rf_model_loaded = False
                # Initialize empty buffers to prevent AttributeErrors later if code tries to access them

        # ----- set up ApproximateTimeSynchronizer subscribers for RF feature alignment -----
        if RF_CONTROLLER and self.rf_model_loaded:
            try:
                qos = QoSProfile(depth=10)
                # message_filters Subscriber requires node and qos_profile kwarg in ROS2 wrapper
                self._mf_flex_sub = Subscriber(self, Float32MultiArray, "/flex_sensor_data", qos_profile=qos)
                self._mf_pressure_sub = Subscriber(self, Float32, "/vacuum_pressure", qos_profile=qos)
                self._mf_force_sub = Subscriber(self, WrenchStamped, "/force_torque_sensor_broadcaster/wrench", qos_profile=qos)
                self._mf_tof_sub = Subscriber(self, Int32, "/tof_sensor_data", qos_profile=qos)

                # tune slop to match sensor skew; 0.05 (50 ms) is a good starting point
                self._rf_sync = ApproximateTimeSynchronizer(
                    [self._mf_flex_sub, self._mf_pressure_sub, self._mf_force_sub, self._mf_tof_sub],
                    queue_size=10,
                    slop=0.05,
                    allow_headerless=True
                )
                self._rf_sync.registerCallback(self.rf_sync_cb)
                self.get_logger().info("RF ApproximateTimeSynchronizer registered (slop=0.05s).")
            except Exception as e:
                self.get_logger().warn(f"Could not create RF ApproximateTimeSynchronizer: {e}")
            self._rf_sync = None


    # --- SERVICE HANDLERS ---
    # update handle_start to store the timer and avoid creating duplicates
    def handle_start(self, request, response):
        if not self.running:
            self.get_logger().info("start_controller called: starting controller...")
            # ... existing startup code ...
            self.running = True
            self.state = 'approach'
            self.controller = 'default'
            self.get_logger().info("Controller started.")

            # --- AUTO STOP AFTER 20 SECONDS ---
            stop_time = 20.0  # seconds
            self.get_logger().info(f"Controller will auto-stop in {stop_time} seconds")

            # If for some reason a leftover timer exists, destroy it first
            if self._auto_stop_timer is not None:
                try:
                    self.destroy_timer(self._auto_stop_timer)
                except Exception:
                    pass
                self._auto_stop_timer = None

            # store the timer so we can cancel/destroy it later
            self._auto_stop_timer = self.create_timer(stop_time, self._auto_stop_once, callback_group=self.cbgroup)
        else:
            self.get_logger().info("start_controller called but controller already running.")
        return response

    # change _auto_stop_once so it destroys the timer (one-shot behavior)
    def _auto_stop_once(self):
        """Stops the controller automatically (called by timer)."""
        self.get_logger().info("Auto-stop timer triggered.")
        try:
            # safe stop
            req = Empty.Request()
            self.handle_stop(req, None)
        except Exception as e:
            self.get_logger().debug(f"_auto_stop_once: error calling handle_stop: {e}")

        # destroy the timer so it doesn't keep firing
        if self._auto_stop_timer is not None:
            try:
                self.destroy_timer(self._auto_stop_timer)
            except Exception as e:
                self.get_logger().debug(f"Could not destroy auto-stop timer: {e}")
            self._auto_stop_timer = None

    # ensure handle_stop also clears/destroys the timer when stopping manually
    def handle_stop(self, request, response):
        if self.running:
            self.get_logger().info("stop_controller: stopping controller, publishing zero twist, and turning off vacuum...")
            self.running = False
            # ... existing shutdown actions ...

            # destroy any pending auto-stop timer
            if self._auto_stop_timer is not None:
                try:
                    self.destroy_timer(self._auto_stop_timer)
                except Exception:
                    pass
                self._auto_stop_timer = None

            self.get_logger().info("Controller stopped and vacuum disabled.")
        else:
            self.get_logger().info("stop_controller called but controller already stopped.")
        return response


    # --- SUBSCRIBERS & PUBLISHERS (unchanged) ---
    def flex_callback(self, msg):
        vals = np.array(msg.data) / 4.0
        self.latest_flex = vals.reshape((4, 1))

    def tof_callback(self, msg):
        self.latest_tof = msg.data
        self.tof_history.append(self.latest_tof)
        self.get_logger().debug(f"ToF history: {list(self.tof_history)}")

    def get_tof_diff(self):
        if len(self.tof_history) < 10:
            return None
        first_avg = sum(list(self.tof_history)[:5]) / 5
        last_avg = sum(list(self.tof_history)[-5:]) / 5
        return last_avg - first_avg

    def pressure_callback(self, msg):
        self.latest_pressure = msg.data  # store suction pressure
    
    def force_callback(self, msg):
        f = msg.wrench.force
        # scalar magnitude
        self.latest_force = (f.x**2 + f.y**2 + f.z**2)**0.5



    def rf_sync_cb(self, flex_msg, pressure_msg, force_msg, tof_msg):
        """
        Called when flex, pressure, force, tof messages are approximately time-aligned.
        Builds the same sliding-window flattened feature vector used at training time.
        Stores the last valid built window in self._last_built_X for _rf_features() to return.
        """
        # 1) compute force magnitude and 21-sample filtered force (training used filter_force(...,21))
        try:
            f = force_msg.wrench.force
            force_mag = float((f.x**2 + f.y**2 + f.z**2)**0.5)
        except Exception as e:
            self.get_logger().debug(f"rf_sync_cb: bad force_msg: {e}")
            return

        # maintain 21-sample buffer and compute simple mean as proxy for filter_force(...,21)
        self._force_raw_buffer.append(force_mag)
        filtered_force = float(np.mean(list(self._force_raw_buffer)))

        # 2) flex norm (training used a single flex_norm per timestep)
        try:
            flex_arr = np.asarray(flex_msg.data, dtype=float) / 4.0
            flex_norm = float(np.linalg.norm(flex_arr))
        except Exception:
            # fallback: try ravel
            try:
                flex_norm = float(np.linalg.norm(np.ravel(np.array(flex_msg.data, dtype=float))))
            except Exception as e:
                self.get_logger().debug(f"rf_sync_cb: cannot parse flex_msg: {e}")
                return

        # 3) pressure and tof scalars
        try:
            pressure = float(pressure_msg.data)
        except Exception:
            pressure = float(getattr(pressure_msg, "data", 0.0))

        try:
            tof = float(tof_msg.data)
        except Exception:
            tof = float(getattr(tof_msg, "data", 0.0))

        # 4) append to sliding windows (order must match training bundle)
        # ensure keys exist (defensive)
        for k in self.rf_feature_order:
            if k not in self._rf_hist:
                self._rf_hist[k] = deque(maxlen=self.rf_window_size)

        self._rf_hist["Flex"].append(flex_norm)
        self._rf_hist["Pressure"].append(pressure)
        self._rf_hist["Force"].append(filtered_force)
        self._rf_hist["TOF"].append(tof)

        # 5) if we have enough samples, build flattened window oldest->newest per sensor
        W = int(self.rf_window_size)
        if all(len(self._rf_hist[k]) >= W for k in self.rf_feature_order):
            feat_list = []
            for k in self.rf_feature_order:
                hist = list(self._rf_hist[k])
                # use most recent W samples in order oldest -> newest
                feat_list.extend(hist[-W:])
            X = np.array(feat_list, dtype=float).reshape(1, -1)

            # optional scaler
            if self.rf_scaler is not None and hasattr(self.rf_scaler, "transform"):
                try:
                    X = self.rf_scaler.transform(X)
                except Exception as e:
                    self.get_logger().warn(f"rf_sync_cb: scaler.transform failed: {e}")
                    X = None

            # sanity-check vs model
            if X is not None and hasattr(self.rf_model, "n_features_in_"):
                if X.shape[1] != self.rf_model.n_features_in_:
                    self.get_logger().warn(
                        f"rf_sync_cb: built {X.shape[1]} features but model expects {self.rf_model.n_features_in_}"
                    )
                    X = None

            # store for control loop to consume
            self._last_built_X = X
        else:
            # not enough history yet
            self._last_built_X = None

    def _rf_features(self):
        """
        Return the most recent aligned window built by rf_sync_cb (or None).
        This preserves your control_loop usage: call _rf_features() and get (1, N) array or None.
        """
        if not getattr(self, "rf_model_loaded", False):
            return None
        return getattr(self, "_last_built_X", None)



    def control_loop(self):
        # --- EARLY EXIT WHEN STOPPED ---
        if not self.running:
            # do not publish or actuate if not running
            return

        diff = self.get_tof_diff()

        if diff is not None:
            # self.get_logger().info(f"ToF change over buffer: {diff}")
            pass

        self.get_logger().info(f"Running?: {self.running}, CONTROLLER: {self.controller}, STATE: {self.state}, force: {self.latest_force}, tof: {self.latest_tof}, pressure: {self.latest_pressure}")
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self.prev_time
        self.prev_time = now
        
        if self.latest_flex is None or self.latest_tof is None:
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
                if (ex < self.position_threshold and ey < self.position_threshold) or self.latest_tof <= self.tof_servo_threshold:
                    self.state = 'approach'
            elif self.state == 'approach':
                if FLEX_CONTROLLER and self.latest_tof > self.tof_servo_threshold and (ex > self.position_threshold or ey > self.position_threshold):
                        self.state = 'servo'
                elif self.latest_tof <= self.tof_relative_motion_threshold:
                    if TOF_CONTROLLER:
                        self.controller = 'relative_controller'
                    else:
                        self.get_logger().info("TOF CONTROLLER IS NOT ENABLED, OPEN LOOP PICK")
                        # --- TOF_CONTROLLER is False: Open-Loop Pick ---
                        self.controller = 'relative_controller'
                        self.get_logger().info("ToF Controller OFF: triggering open-loop pick")
                        self.state = 'pick'
                        self.pick_start_time = now
                        self.latest_pressure = None
                        self.get_logger().info(f'Picking: turning on vacuum (tof = {self.latest_tof})')
                        self.pump.vacuum_on()
        if self.controller == 'relative_controller':
            if self.state == 'servo':
                # Switch to 'approach' if centered OR below servo threshold
                if (ex < self.position_threshold and ey < self.position_threshold) or self.latest_tof <= self.tof_servo_threshold:
                    self.state = 'approach'
            if self.state == 'approach':
                if ex > self.position_threshold or ey > self.position_threshold:
                    if FLEX_CONTROLLER:
                        self.state = 'servo'
                    elif not FLEX_CONTROLLER:
                        self.get_logger().info("FLEX CONTROLLER IS NOT ENABLED STAY IN APPROACH STATE")
                if self.get_tof_diff() < 0: # apple is getting closer
                    self.get_logger().info("apple is getting closer")
                elif self.get_tof_diff() > 1: # apple is being pushed away
                    self.get_logger().info("apple is getting pushed away")
                    self.state = 'reverse'
                else: # apple is nicely aligned
                    self.get_logger().info("apple is nicely aligned")
                    self.state = 'pick'
                    self.pick_start_time = now
                    self.latest_pressure = None
                    self.get_logger().info(f'Picking: turning on vacuum (tof = {self.latest_tof})')
                    self.pump.vacuum_on()
            if self.state == 'reverse':
                if self.get_tof_diff() < 0: # apple is getting closer
                    self.get_logger().info("apple is getting closer")
                elif self.get_tof_diff() > 1: # apple is being pushed away
                    self.get_logger().info("apple is getting further away")
                    self.state = 'approach'
                else: # apple is nicely aligned
                    self.get_logger().info("apple is nicely aligned")
                    self.state = 'pick'
                    self.pick_start_time = now
                    self.latest_pressure = None
                    self.get_logger().info(f'Picking: turning on vacuum (tof = {self.latest_tof})')
                    self.pump.vacuum_on()
            elif self.state == 'pick':
                # immediate brake:
                self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.00
                elapsed = now - self.pick_start_time # start of grasp
                pressure = self.latest_pressure if self.latest_pressure is not None else float('inf')
                self.get_logger().info(f"[DEBUG] pick elapsed={elapsed:.2f}, pressure={pressure}")
                # Success
                if PRESSURE_CONTROLLER:
                    if pressure <= PRESSURE_THRESHOLD:
                        self.get_logger().info(f'Grasp vacuum succeeded (pressure={pressure})')
                        self.state = 'release' # move on to the picking motion
                        if RF_CONTROLLER and self.rf_model_loaded:
                            for key in self._rf_hist:
                                self._rf_hist[key].clear()
                        self.release_start_time = now
                    elif elapsed > PRESSURE_CONTROLLER_TIMEOUT:
                        self.get_logger().warn(f'Grasp failed: timeout (pressure={pressure})')
                        self.pump.vacuum_off()
                        self.state = 'failed'  # use a failure state instead of immediate shutdown
                # Timeout
                elif elapsed > PRESSURE_CONTROLLER_TIMEOUT:
                    self.get_logger().warn(f'Grasp completed: timeout (pressure={pressure})')
                    self.state = 'release' # move on to the picking motion
                    if RF_CONTROLLER and self.rf_model_loaded:
                        for key in self._rf_hist:
                            self._rf_hist[key].clear()
                    self.release_start_time = now
            elif self.state == 'release':
                elapsed_release = now - self.release_start_time
                self.get_logger().info(f"RELEASE elapsed={elapsed_release:.2f}")

                if RF_CONTROLLER and self.rf_model_loaded:
                    self.get_logger().info("RF CONTROLLER STUFF IS HAPPENING NOW")
                    if now - self.last_rf_time > self.rf_period:
                        self.last_rf_time = now

                        features = self._rf_features()
                        if features is not None:
                            assert features.shape[1] == self.rf_model.n_features_in_

                            label = int(self.rf_model.predict(features)[0])
                            self.get_logger().info(f"RF predicted label={label}")

                            if label == 1:
                                self.get_logger().info("RF SUCCESS → done")
                                self.state = "done"
                                return

                            elif label in (2, 3):
                                self.get_logger().warn("RF FAILURE → abort")
                                self.pump.vacuum_off()
                                self.state = "failed"
                                return

                # Pulling back (picking motion)
                if elapsed_release < PICKING_TIME:
                    self.get_logger().info("retreating...")
                # Release
                elif elapsed_release < RELEASE_TIME:
                    self.get_logger().info("releasing apple now...")
                    self.pump.vacuum_off()

                # Final state resolution
                else:
                    self.get_logger().info("Entire controller done")
                    self.state = 'done'

        cmd_wz = 0.0   # default: no rotation
        # --- Command selection ---
        if self.state == 'servo':
            cmd_vx, cmd_vy, cmd_vz = -vx, -vy, 0.1
        elif self.state == 'approach':
            cmd_vx, cmd_vy = 0.0, 0.0
            cmd_vz = 0.1 * self.velocity_scale_factor_z
        elif self.state == 'reverse':
            cmd_vx, cmd_vy = 0.0, 0.0
            cmd_vz = -0.1 * self.velocity_scale_factor_z
        elif self.state == 'release':
            elapsed_release = now - self.release_start_time
            if elapsed_release < 2.0:
                cmd_vx, cmd_vy, cmd_vz = 0.0, 0.0, -2.0
                cmd_wz = -4.0
            elif elapsed_release < 4.0:
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
        # self.get_logger().info(f"PUBLISHING COMMAND!!!!: {cmd}")

        # Debug apple pos
        apple = Float32MultiArray(data=[float(self.x[1]), float(self.x[0])])
        self.apple_pub.publish(apple)


    # The rest of your helper functions are unchanged; include them as-is:
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
        self.alpha_pos = 0.9
        self.alpha_cmd = 0.3
        self.K_p = 0.3
        self.K_i = 0.0
        self.K_d = 0.01
        self.integral_x = self.integral_y = 0.0
        self.prev_err_x = self.prev_err_y = 0.0
        self.vel_max = 0.3
        self.acc_max = 3.0

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

    def _enable_servo_mode(self, frame: str = "tool0", sim=False):
        req = SwitchController.Request()
        req.activate_controllers = ["forward_position_controller"]
        if sim:
            req.deactivate_controllers = ["scaled_joint_trajectory_controller"]
        else:
            req.deactivate_controllers = ["scaled_joint_trajectory_controller"]
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
        if hasattr(node, "pump"):
            node.get_logger().info("Turning off vacuum before exit...")
            node.pump.vacuum_off()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
