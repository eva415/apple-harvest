# publish vacuum levels
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from eva_vacuum_test import PumpIO  # your vacuum control file

class VacuumPublisher(Node):
    def __init__(self):
        super().__init__('vacuum_publisher')

        # Publisher
        self.pressure_pub = self.create_publisher(Float32, 'vacuum_pressure', 10)

        # Timer settings
        self.control_period = 0.01  # 100 Hz

        # Initialize Pump
        self.pump = PumpIO(self)

        # Timer for periodic publishing
        self.create_timer(self.control_period, self.publish_pressure)

    def publish_pressure(self):
        vacuum_level = self.pump.read_vacuum()
        if vacuum_level is not None:
            msg = Float32()
            msg.data = float(vacuum_level)  # ensure it’s a float
            self.pressure_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VacuumPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
