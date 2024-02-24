import math
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
        print('pos x',msg.transforms[0].transform.translation.x,
              ', y:', msg.transforms[0].transform.translation.y,
              ', z:', msg.transforms[0].transform.translation.z)

        qx = msg.transforms[0].transform.rotation.x
        qy = msg.transforms[0].transform.rotation.y
        qz = msg.transforms[0].transform.rotation.z
        qw = msg.transforms[0].transform.rotation.w
        print('Quat x', qx,
              ', y:', qy,
              ', z:', qy,
              ', w:', qw)
        
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)
        print('roll:', roll, ', pitch:', pitch, ', yaw:', yaw)


                        
        #exit()


def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
        
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

