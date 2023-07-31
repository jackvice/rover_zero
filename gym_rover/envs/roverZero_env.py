import gym
import rclpy
from rclpy.node import Node

import logging


import numpy as np
import math
import time
import os
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf2_msgs.msg import TFMessage

#from gym import spaces
from gym import utils, spaces
from gym.utils import seeding
#from gym_rover.mytf.transformations import euler_from_quaternion
from gym_rover.mytf import euler_from_quaternion

#from gym_rover.envs import Respawn


class RoverZeroEnv(gym.Env):

    def __init__(self, 
            goal_list=None,
            max_env_size=None,
            continuous=False,
            observation_size = 6, 
            action_size=2, 
            min_range = 0.13,
            max_range = 3.5,
            min_ang_vel = -1.5,
            max_ang_vel = 1.5,
            const_linear_vel = 0.15,
            goalbox_distance = 0.35,
            collision_distance = 0.13,
            reward_goal=200,
            reward_collision=-200,
            angle_out = 135
        ):

        self.max_episode_steps = 100
        self.goal_x = 0
        self.goal_y = 0
        #self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()

        # Create the node after the new ROS_DOMAIN_ID is set in generate_launch_description()
        rclpy.init()
        self.node = rclpy.create_node(self.__class__.__name__)

        
        #self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.pub_cmd_vel = self.node.create_publisher(Twist, 'cmd_vel', 10)
                                               #qos_profile=qos_profile_sensor_data)

        self.sub_pose = self.node.create_subscription(TFMessage, # position and orientation
                                                      '/world/maze/dynamic_pose/info',
                                                      self.observation_callback, 10)
                                                      #qos_profile=qos_profile_sensor_data)
                                                      
        # ahole, is this needed.                                              
        self.reset_sim = self.node.create_client(Empty, '/reset_simulation')
        
        #self.subscription = self.create_subscription(
        #    TFMessage,
        #    'world/maze/dynamic_pose/info',
        #    self.listener_callback,
        #    10)


        #self.respawn_goal.setGoalList(goal_list)

        self.observation_size = observation_size
        self.const_linear_vel = const_linear_vel
        self.min_range = min_range
        self.max_range = max_range
        self.min_ang_vel = min_ang_vel
        self.max_ang_vel = max_ang_vel
        self.goalbox_distance = goalbox_distance
        self.collision_distance = collision_distance
        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.angle_out = angle_out
        self.continuous = continuous
        self.max_env_size = max_env_size

        self.reward_target = [-4.0, 3.0] # x and y

        self.old_distance = 5


        #low, high, shape_value = self.get_action_space_values()
        #self.action_space = spaces.Box(low=low, high=high, shape=(shape_value,), dtype=np.float32)

        self.action_space = spaces.Box(
            np.array([-1.0, -3.0]).astype(np.float32),
            np.array([1.0, 3.0]).astype(np.float32),
        )


        low, high = self.get_observation_space_values()

        # x,y,z, roll, pitch, yaw
        self.observation_space = spaces.Box(low=np.array([-15.0, -15.0,-15.0, -180.0, -180.0, -180.0,]),
                                            high=np.array([15.0, 15.0, 15.0, 180.0, 180.0, 180.0,]), 
                                            shape=(6, ), dtype=np.float32 )

        self.num_timesteps = 0
        self.lidar_distances = None
        self.ang_vel = 0

        self.start_time = time.time()
        self.last_step_time = self.start_time
        #self.seed()
        
    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        #print(message)
        #exit()
        self.observation_pose_msg = message
        
    def take_observation(self):
        """
        Take observation from the environment and return it.
        :return: state.
        """
        # # # # Take an observation
        rclpy.spin_once(self.node)
        #obs_message = self.observation_pose_msg
        # Check that the observation is not prior to the action
        # obs_message = self._observation_msg
        #while obs_message is None or int(str(self.observation_pose_msg.header.stamp.sec)+
        #                                 (str(self.observation_pose_msg.header.stamp.nanosec))) < self.ros_clock:
        #    # print("I am in obs_message is none")
        #    rclpy.spin_once(self.node)
        #    obs_message = self.observation_pose_msg

        #print('pos z',msg.transforms[0].transform.translation.z)


        #self.position = obs_message.transforms[0].transform.translation
        z_position = self.observation_pose_msg.transforms[0].transform.translation
        orientation = self.observation_pose_msg.transforms[0].transform.rotation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        #goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)    
        
        return np.array( [z_position.x, z_position.y, z_position.z, roll, pitch, yaw] ).astype(np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.num_timesteps += 1

        #self.set_ang_vel(action)

        vel_cmd = Twist()
        #vel_cmd.linear.x = 0.05 #self.const_linear_vel # ahole
        #vel_cmd.angular.z = 0.0 #self.ang_vel # ahole

        vel_cmd.linear.x = float(np.clip(action[0], -1, 1))
        vel_cmd.angular.z = float(np.clip(action[1], -3, 3))
        
        self.pub_cmd_vel.publish(vel_cmd)
        self.ros_clock = rclpy.clock.Clock().now().nanoseconds
        
        new_state = self.take_observation()

        #reward = 1.0 #self.setReward(state, done, action)
        #reward = self.setReward(state, done, action)
        
        reward = self.navigationReward(new_state)
        done = bool(self.num_timesteps == self.max_episode_steps)
        #print("step about to return ", np.asarray([new_state]), np.asarray(reward), done, False, {})
        #exit() np.array([-1.0, -3.0]).
        return new_state, reward, done, False, {}

    def navigationReward(self, state): #        x                  y
        new_distance = math.dist((self.reward_target[0], self.reward_target[1] ), ( state[0], state[1] ) )
        #print("new_distance", new_distance, "old_distance", self.old_distance)

        if new_distance <= .05: # goal
            print("we did it!")
            exit()
        if state[3] < -3.14: # flip
            print("flipped ")
            exit()
        if self.old_distance - new_distance == 0:
            reward = 0
        else: #                                                           Roll punishment
            reward = (0.0005 / (self.old_distance - new_distance)) - (100 * abs(state[4])) 
        #print("reward ", reward, ",   (100 * abs(state[3])", 100 * abs(state[3]) )
        #exit()        
        #logger.info("old", self.old_distance,", new", new_distance)

        self.old_distance = new_distance

        return reward

    #def get_action_space_values(self):
    #    lin_low = -3.0 #self.min_ang_vel
    #    lin_high = 3.0 #self.max_ang_vel
    #    ang_low = -3.0 #self.min_ang_vel
    #    ang_high = 3.0 #self.max_ang_vel
    #    shape_value = 2
    #    return low, high, shape_value

    
    def get_observation_space_values(self):
        low = np.append(np.full(self.observation_size, self.min_range), np.array([-math.pi, 0], dtype=np.float32))
        high = np.append(np.full(self.observation_size, self.max_range), np.array([math.pi, self.max_env_size], dtype=np.float32))
        return low, high

    def reset(self):
        
        #Reset the agent for a particular experiment condition.
        
        self.num_timesteps = 0
        """
        if True: #self.reset_jnts is True:
            # reset simulation
            while not self.reset_sim.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('/reset_simulation service not available, waiting again...')

            reset_future = self.reset_sim.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_future)
        """
        self.ros_clock = rclpy.clock.Clock().now().nanoseconds

        # Take an observation
        obs = self.take_observation()

        # Return the corresponding observation
        return obs, {}

    
    def render(self, mode=None):
        pass


    def close(self):
        self.reset()






        
    """
    def _getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance
    
    def getOdometry(self, odom):
        #self.position = odom.pose.pose.position
        #orientation = odom.pose.pose.orientation

        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = heading
    
    def get_time_info(self):
        time_info = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
        time_info += '-' + str(self.num_timesteps)
        return time_info

    def episode_finished(self):
        pass

    
    #def preprocess_lidar_distances(self, scan_range): # ahole, might need for later
    #    return scan_range

    #def get_env_state(self):
    #    return self.lidar_distances

    
    
    def getState(self, scan):
        scan_range = []
        heading = self.heading
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(self.max_range)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(self.min_range)
            else:
                scan_range.append(scan.ranges[i])

        self.lidar_distances = self.preprocess_lidar_distances(scan_range)
        time_info = self.get_time_info()
        current_distance = self._getGoalDistace()

        if min(self.lidar_distances) < self.collision_distance:
            print(f'{time_info}: Collision!!')
            done = True

        if abs(heading) > math.radians(self.angle_out):
            print(f'{time_info}: Out of angle')
            done = True

        if current_distance < self.goalbox_distance:
            if not done:
                print(f'{time_info}: Goal!!')
                self.get_goalbox = True
                if self.respawn_goal.last_index is (self.respawn_goal.len_goal_list - 1):
                    done = True
                    self.episode_finished()
            
        return self.get_env_state() + [heading, current_distance], done
 

    def navigationReward(self, heading):
        reference = 1-2*abs(heading)/math.pi
        reward = 5*(reference ** 2)

        if reference < 0:
            reward = -reward

        return reward


    def setReward(self, state, done, action):
                
        if self.get_goalbox:
            reward = self.reward_goal
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True)
            self.goal_distance = self._getGoalDistace()
            self.get_goalbox = False

        elif done:
            reward = self.reward_collision=-200
            self.pub_cmd_vel.publish(Twist())
            if self.respawn_goal.last_index != 0:
                self.respawn_goal.initIndex()
                self.goal_x, self.goal_y = self.respawn_goal.getPosition()
                self.goal_distance = self._getGoalDistace()
        
        else:
            heading = state[-2]
            reward = self.navigationReward(heading)

        return reward

    def set_ang_vel(self, action):
        if self.continuous:
            self.ang_vel = action
        else:
            self.ang_vel = self.actions[action]
            

    def stepOLd(self, action):

        self.set_ang_vel(action)

        vel_cmd = Twist()
        vel_cmd.linear.x = self.const_linear_vel
        vel_cmd.angular.z = self.ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)
        self.num_timesteps += 1

        return np.asarray(state), reward, done, {}



    def resetTurtle(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
            time.sleep(1)

        self.goal_distance = self._getGoalDistace()
        state, _ = self.getState(data)

        return np.asarray(state)
    """
    
