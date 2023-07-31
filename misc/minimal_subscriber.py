import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        #self.subscription = self.create_subscription(
        #    String,
        #    'topic',
        #    self.listener_callback,
        #    10)
        self.subscription = self.create_subscription(
            TFMessage,
            'world/maze/dynamic_pose/info',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        #self.get_logger().info('I heard: "%s"' % msg.transforms[0].transform.translation.x)
        #print(dir(msg.transforms[0].transform.translation))
        #print('pos x',msg.transforms[0].transform.translation.x)
        #print('pos y',msg.transforms[0].transform.translation.y)
        #print('pos z',msg.transforms[0].transform.translation.z)
        print('quat x',msg.transforms[0].transform.rotation.x)
        print('quat y',msg.transforms[0].transform.rotation.y)
        print('quat z',msg.transforms[0].transform.rotation.z)
        print('quat w',msg.transforms[0].transform.rotation.w)


                        
        #exit()


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

