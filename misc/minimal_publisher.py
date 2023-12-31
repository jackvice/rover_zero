# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing mara joint angles.
#from control_msgs.msg import JointTrajectoryControllerState
# from gazebo_msgs.srv import SetEntityState, DeleteEntity
#from gazebo_msgs.msg import ContactState, ModelState#, GetModelList
from std_msgs.msg import String
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist
from ros2pkg.api import get_prefix_path
from builtin_interfaces.msg import Duration

#exit()

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'jackstring', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        self.publisher_v = self.create_publisher(Twist, 'cmd_vel', 10)
        #self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        
    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

        vel_cmd = Twist()
        vel_cmd.linear.x = .2 #self.const_linear_vel 
        vel_cmd.angular.z = -0.00 #self.ang_vel
        self.publisher_v.publish(vel_cmd)
        #self.pub_cmd_vel.publish(vel_cmd)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
