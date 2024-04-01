import gym
import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import os
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gym import spaces
from gym.utils import seeding
from gym_turtlebot3.envs.mytf import euler_from_quaternion
from gym_turtlebot3.envs import Respawn
import random

class TurtleBot3Env(gym.Env, Node):
    def __init__(self,
                 goal_list=None,
                 max_env_size=None,
                 continuous=True,
                 observation_size=24,
                 action_size=5,
                 min_range=0.01, #0.13
                 max_range=3.9, #lidar max range is 3.8
                 min_ang_vel=-0.5, #-1.5,
                 #max_ang_vel=1.5,
                 max_ang_vel=0.5, #1.5,
                 min_linear_vel=-0.5,
                 max_linear_vel=0.8,
                 const_linear_vel=0.04, #0.15
                 goalbox_distance=0.35,
                 collision_distance=0.13,
                 reward_goal=200,
                 reward_collision=-200,
                 angle_out=135):
        print("gym.Env)", gym.Env)
        Node.__init__(self, 'turtlebot3_env_node')
        gym.Env.__init__(self)

        # Assigning class parameters to the values passed in or to their default values
        self.goal_list = goal_list if goal_list is not None else []
        self.max_env_size = max_env_size
        self.continuous = continuous
        self.num_lidar = 360
        self.max_timesteps = 22000000 # ~ 3 minutes
        self.observation_size = 362 # 360 lidar + heading and distance
        self.action_size = action_size
        self.min_range = min_range
        self.max_range = max_range
        self.min_ang_vel = min_ang_vel
        self.max_ang_vel = max_ang_vel
        self.min_linear_vel = min_linear_vel
        self.max_linear_vel = max_linear_vel
        self.const_linear_vel = const_linear_vel
        self.goalbox_distance = goalbox_distance
        self.collision_distance = collision_distance
        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.angle_out = angle_out
        
        self.goal_x = 0.0  # Default value
        self.goal_y = 0.0  # Default value

        self.latest_scan = None
        self.new_scan_received = False


        # Initializing ROS2 publishers, subscribers, and service clients
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 5)
        self.sub_odom = self.create_subscription(Odometry, 'odom', self.getOdometry, 10)
        self.reset_client = self.create_client(Empty, 'gazebo/reset_simulation')
        self.unpause_client = self.create_client(Empty, 'gazebo/unpause_physics')
        self.pause_client = self.create_client(Empty, 'gazebo/pause_physics')

        # Additional environment setup
        self.respawn_goal = Respawn()
        print('init self.goal_list', self.goal_list)
        #exit()
        self.respawn_goal.setGoalList(self.goal_list)
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.sub_scan = self.create_subscription(LaserScan, 'scan', self.laser_scan_callback, 10)

        # Defining the action and observation spaces based on the environment parameters

        low, high, shape_value = self.get_action_space_values()
        self.action_space = spaces.Box(low=low, high=high, shape=(shape_value,), dtype=np.float32)

        
        low, high = self.get_observation_space_values()

        # Now define the observation space with these bounds
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Other initializations
        self.num_timesteps = 0
        self.lidar_distances = None
        self.ang_vel = 0.0
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.seed()

    def step(self, action):
        # Set the robot's angular velocity based on the action
        self.ang_vel = action
        print(action)
        # Publish the velocity command
        vel_cmd = Twist()
        vel_cmd.linear.x = self.const_linear_vel
        vel_cmd.angular.z = float(self.ang_vel)
        self.pub_cmd_vel.publish(vel_cmd)

        # Wait for the next laser scan data to be received
        # Assuming `self.new_scan_received` is a flag that's True when new data is received
        # and reset to False after being processed
        scan_data = self.wait_for_scan_data()
        
        # Now, `self.lidar_distances` should contain the latest processed scan data
        # Reset the flag to False to prepare for the next step
        self.new_scan_received = False

        # Use the latest scan data to determine the state and whether the episode is done
        state, done = self.getState()
        info = {}
        if done:
            if self.get_goalbox:
                info['termination_reason'] = 'goal_reached'
            elif min(scan_data) < self.collision_distance:
                info['termination_reason'] = 'collision'
            elif self.num_timesteps >= self.max_timesteps:
                info['termination_reason'] = 'time_limit'
            else:
                info['termination_reason'] = 'other'

        # Calculate the reward based on the state and whether the episode is done
        reward = self.setReward(state, done, action)
        
        # Increment the timestep counter
        self.num_timesteps += 1

        # Return the current state, reward, done flag, and an empty info dict
        if not self.observation_space.contains(state):
            print("Observation out of bounds with length:",len(state), ", and values:", state)
            exit()
        return np.asarray(state), reward, done, False, info # false for truncated

    
    def getState(self):
        if self.latest_scan is None:
            #print("latest_scan is None: using last known scan")
            scan_range = self.last_known_scan  # Use the last known scan
        else:
            scan_range = self.latest_scan
            self.last_known_scan = scan_range  # Update the last known scan
        heading = self.heading
        done = False

        # Initialize your state array or list
        state = []

        # Add the processed lidar distances to the state
        state.extend(scan_range)

        # Add heading and current distance to the goal to the state
        # Assuming self._getGoalDistance() calculates the distance to the goal
        current_distance = self._getGoalDistace()
        state.append(heading)
        state.append(current_distance)

        # Check for collision
        if min(scan_range) < self.collision_distance:
            #print(f'Collision detected')
            done = True

        # Check if the robot is out of angle bounds
        #print('heading', abs(heading),'angle_out', math.radians(self.angle_out))
        #if abs(heading) > math.radians(self.angle_out):
        #    print(f'Out of angle bounds')
        #    done = True

        # Check for goal achievement
        if current_distance < self.goalbox_distance:
            print(f'Goal achieved')
            self.get_goalbox = True
            done = True
        return np.array(state, dtype=np.float32), done

    def wait_for_scan_data(self):
        # Wait for the next laser scan data to be received
        while not self.new_scan_received:
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks to receive new messages.
    
        # At this point, self.new_scan_received is True, meaning new data is available.
        # Reset the flag for the next wait cycle.
        self.new_scan_received = False
        # Return the latest scan data for further processing.
        return self.latest_scan # list of 360 values 

    def get_observation_space_values(self):
        # Assuming min_range and max_range define the bounds for LIDAR values,
        # and the heading is between -pi and pi, and distance between some min and max value.
        # Let's say distance can vary from 0 to some max_distance you define based on your environment.

        max_distance = 10  # maximum distance, adjust based on your environment
 
        # Update the low and high bounds to include LIDAR, heading, and distance
        low = np.array([self.min_range] * 360 + [-math.pi, 0], dtype=np.float32)
        high = np.array([self.max_range] * 360 + [math.pi, max_distance], dtype=np.float32)

        return low, high

    def _getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

    def generate_new_goal(self):
        # Example implementation - generate random goal coordinates within a defined range
        self.goal_x = random.uniform(-self.max_env_size, self.max_env_size)
        self.goal_y = random.uniform(-self.max_env_size, self.max_env_size)
        return self.goal_x, self.goal_y

    
    def getOdometry(self, odom):
        if not hasattr(self, 'goal_x') or not hasattr(self, 'goal_y'):
            # Handle the case where goal_x or goal_y is not yet defined
            # This might involve setting a default value, or skipping certain computations
            return
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

    def preprocess_lidar_distances(self, scan_range):
        return scan_range

    def get_env_state(self):
        return self.lidar_distances


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

    def call_service(self, client, request):
        """
        Call a ROS 2 service and wait for the response.
        
        :param client: The service client to use for the call.
        :param request: The service request message.
        """
        # Ensure the client is ready with a timeout
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f'Service {client.srv_name} not available')
            return None
        
        # Call the service and wait for the response
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, executor=SingleThreadedExecutor())
        
        if future.result() is not None:
            return future.result()
        else:
            self.get_logger().error('Service call failed')
            return None

    def reset(self):
        #print('\n\n\n    In reset()')
        # Generate a new goal for the TurtleBot
        self.goal_x, self.goal_y = self.generate_new_goal()

        # Instead of resetting the simulation or robot position, 
        # just ensure that any necessary environment variables are re-initialized as needed
        # This could include clearing or updating goal-related flags and counters
        self.new_scan_received = False
        self.initGoal = False  # Assuming initGoal is used to track goal initialization status
        
        # Wait for the latest sensor data to ensure the environment is ready
        # This could be immediately after a goal is reached or any other condition that
        # you define as requiring a 'reset'
        self.wait_for_scan_data()
        
        # Once the new goal is set and the latest scan data is available, calculate the initial state
        self.goal_distance = self._getGoalDistace()
        state, _ = self.getState()  
        
        state = np.array(state, dtype=np.float32)
        
        # Debugging prints can be adjusted or removed as necessary
        #print(f"Reset state shape: {state.shape}")
        #print("Reset state type", type(state))
        #print("Reset state [0]", state[0])
        #exit()
        # Check that the state is within the observation space bounds
        assert self.observation_space.contains(state), "Observation out of bounds"
        
        # Return the initial state and an empty dictionary, similar to a traditional reset
        #but without restarting the environment
        return state, {}


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_default_state(self):
        # Return a default state that matches the expected shape and type
        # of your environment's observation space.
        # This is just an example; adjust according to your observation space.
        default_state = [0] * self.observation_size  # Assuming self.observation_size is defined
        return default_state

    
    def get_action_space_values(self):
        low = self.min_ang_vel
        high = self.max_ang_vel
        shape_value = 1

        return low, high, shape_value

    def laser_scan_callback(self, msg):
        # Assuming msg.ranges contains the distances
        processed_scan = [min(distance, self.max_range) if not np.isinf(distance) and
                          not np.isnan(distance) else self.max_range for distance in msg.ranges]
        self.latest_scan = processed_scan  # This should ensure latest_scan is never None when used
        self.new_scan_received = True


    def render(self, mode=None):
        pass


    def close(self):
        self.reset()
