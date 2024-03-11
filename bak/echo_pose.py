import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class EchoPoseNode(Node):
    def __init__(self):
        super().__init__('echo_pose_node')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription  # prevent unused variable warning

    def odom_callback(self, msg):
        self.get_logger().info('Pose: {}'.format(msg.pose.pose))

def main(args=None):
    rclpy.init(args=args)
    echo_pose_node = EchoPoseNode()
    rclpy.spin(echo_pose_node)
    echo_pose_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
