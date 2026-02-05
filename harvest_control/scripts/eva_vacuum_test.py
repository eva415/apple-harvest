#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ur_msgs.srv import SetIO
from ur_msgs.msg import IOStates
import time

class PumpIO:
    def __init__(self, ros_node: Node):
        self.node = ros_node
        self.DO_VACUUM = 4     # DO4 -> Pump pin 3
        self.DO_BLOWOFF = 5    # DO5 -> Pump pin 4
        self.DO_DISABLE_ES = 6 # DO6 -> Pump pin 8
        self.VAC_SENSOR_PIN = 10 # AI10 -> Pump pin 6

        self.vacuum_voltage = None

        # Subscribe to IOStates to get analog input
        self.node.create_subscription(IOStates, '/io_and_status_controller/io_states', self.io_callback, 10)

        # Service client to set outputs
        self.cli_set = ros_node.create_client(SetIO, '/io_and_status_controller/set_io')
        while not self.cli_set.wait_for_service(timeout_sec=1.0):
            ros_node.get_logger().info('Waiting for /io_and_status_controller/set_io...')

    def io_callback(self, msg: IOStates):
        # Vacuum sensor is AI0 (analog input pin 0)
        for ai in msg.analog_in_states:
            if ai.pin == 0:  # <- Corrected pin number
                self.vacuum_voltage = ai.state  # voltage from sensor

    def set_do(self, pin, state: bool):
        req = SetIO.Request()
        req.fun = 1
        req.pin = pin
        req.state = 1.0 if state else 0.0
        future = self.cli_set.call_async(req)
        # rclpy.spin_until_future_complete(self.node, future)
        # if not future.result().success:
        #     self.node.get_logger().warn(f"Failed to set DO{pin} to {state}")
        # return future.result().success
        return

    def read_vacuum(self):
        """Return vacuum level in -kPa based on last received voltage (1–5V sensor)"""
        if self.vacuum_voltage is None:
            self.node.get_logger().warn("No vacuum sensor data received yet")
            return None
        
        # Clamp voltage to expected range
        voltage = max(1.0, min(5.0, self.vacuum_voltage))
        
        # Convert to vacuum in -kPa
        vacuum_level = ((voltage - 1.0) / 4.0) * 101.3

        return -vacuum_level

    def vacuum_on(self):
        return self.set_do(self.DO_VACUUM, True)

    def vacuum_off_and_blowoff(self):
        self.set_do(self.DO_VACUUM, False)
        self.set_do(self.DO_BLOWOFF, True)
        self.node.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.2))
        self.set_do(self.DO_BLOWOFF, False)

    def vacuum_off(self):
        self.set_do(self.DO_VACUUM, False)
        self.set_do(self.DO_BLOWOFF, False)

    def disable_energy_saving(self):
        return self.set_do(self.DO_DISABLE_ES, True)


def main():
    rclpy.init()
    node = Node("vacuum_test_node")
    pump = PumpIO(node)

    # Give ROS some time to receive analog input
    node.get_logger().info("Waiting for initial vacuum sensor data...")
    for _ in range(20):  # wait up to ~2 sec
        rclpy.spin_once(node, timeout_sec=0.1)
        if pump.vacuum_voltage is not None:
            break
        time.sleep(0.1)

      # Take multiple atmospheric readings before pump start
    samples = []
    node.get_logger().info("Collecting atmospheric vacuum readings (pump OFF)...")
    for _ in range(10):  # take 10 samples over ~1s
        rclpy.spin_once(node, timeout_sec=0.1)
        v = pump.read_vacuum()
        if v is not None:
            samples.append(v)
        time.sleep(0.1)

    if samples:
        baseline = sum(samples) / len(samples)
        node.get_logger().info(f"Atmospheric baseline (pump OFF): {baseline:.2f} kPa")
    else:
        node.get_logger().warn("No atmospheric samples collected")
        
    # Now turn on pump
    pump.vacuum_on()
    node.get_logger().info("Pump turned on. Reading vacuum...")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            vacuum = pump.read_vacuum()
            if vacuum is not None:
                node.get_logger().info(f"Vacuum level: {vacuum:.1f} kPa")
            else:
                node.get_logger().info("Waiting for vacuum sensor data...")
            time.sleep(0.1)

    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        pump.vacuum_off_and_blowoff()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()