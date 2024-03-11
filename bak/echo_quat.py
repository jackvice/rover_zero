import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class QuaternionPrinter(Node):
    def __init__(self):
        super().__init__('quaternion_printer')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription  # prevent unused variable warning

    def odom_callback(self, msg):
        quaternion = msg.pose.pose.orientation
        self.get_logger().info('Quaternion: x=%f y=%f z=%f w=%f' % (quaternion.x, quaternion.y, quaternion.z, quaternion.w))

def main(args=None):
    rclpy.init(args=args)
    quaternion_printer = QuaternionPrinter()
    rclpy.spin(quaternion_printer)
    quaternion_printer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
