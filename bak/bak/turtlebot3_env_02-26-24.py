import gym
import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import os
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gym import spaces
from gym.utils import seeding
from gym_turtlebot3.envs.mytf import euler_from_quaternion
from gym_turtlebot3.envs import Respawn
import random
import time

class TurtleBot3Env(gym.Env, Node):
    def __init__(self, goal_list=None, max_env_size=None, continuous=True):
        #print("gym.Env)", gym.Env)
        Node.__init__(self, 'turtlebot3_env_node')
        gym.Env.__init__(self)
        self.start_time = time.time()

        self.position = Point(x=0.0, y=0.0, z=0.0)
        
        # Assigning class parameters to the values passed in or to their default values
        #self.goal_list = goal_list if goal_list is not None else []
        self.max_env_size = None
        self.continuous = True
        self.num_lidar = 360
        self.max_timesteps = 22_000_000 #
        self.observation_size = 370 # 360 lidar + heading and distance
        self.action_size = 2
        self.min_range = -3.2 #for pi  #0.01, #0.13
        self.max_range = 11 #lidar max range is 3.8
        self.min_ang_vel=-0.5 #-1.5,
        self.max_ang_vel=0.5 #1.5,
        self.min_linear_vel=-0.03
        self.max_linear_vel=0.04
        self.previous_imu_state = None
        #reward shaping
        self.safe_distance=0.3 # 0.3 is 12 inches which is 6" from corner
        self.max_timesteps = 25000

        self.latest_scan = None
        self.new_scan_received = False

        # Initializing ROS2 publishers, subscribers, and service clients
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 5)
        self.reset_client = self.create_client(Empty, 'gazebo/reset_simulation')
        self.sub_scan = self.create_subscription(LaserScan, 'scan',
                                                 self.laser_scan_callback, 10)
        self.sub_scan = self.create_subscription(Imu, 'imu', self.imu_callback, 10)
        
         # Defining the action and observation spaces based on the environment parameters
        low_A, high_A, shape_value = self.get_action_space_values()
        self.action_space = spaces.Box(low=low_A, high=high_A, shape=(shape_value,),
                                       dtype=np.float32)
 
        self.observation_space = self.create_observation_space()
        
        # Other initializations
        self.num_timesteps = 0
        self.lidar_distances = None
        self.linear_vel = 0.0
        self.ang_vel = 0.0

        self.last_step_time = self.start_time

        self.previous_position = None
        #self.stay_place_penalty = -0.01
        self.latest_imu = None
        self.previous_imu = None  # Or a default structure that matches IMU data

    def preprocess_observation(self, obs_dict):
        # Assuming obs_dict is your observation dictionary with 'lidar' and 'imu' keys
        lidar_data = obs_dict['lidar']
        imu_data = obs_dict['imu']

        # Flatten and concatenate the lidar and imu data into a single vector
        flat_obs = np.concatenate([lidar_data.flatten(), imu_data.flatten()])

        # Convert the flat observation array to float32 and add a batch dimension
        flat_obs = np.expand_dims(np.array(flat_obs, dtype=np.float32), axis=0)
    
        return flat_obs

    def step(self, action):
        # Set the robot's angular velocity based on the action
        self.ang_vel = action[0]
        self.linear_vel = action[1]

        # Publish the velocity command
        vel_cmd = Twist()
        vel_cmd.linear.x = float(self.linear_vel) # continuous 
        vel_cmd.angular.z = float(self.ang_vel)
        self.pub_cmd_vel.publish(vel_cmd)

        # Wait for the next laser scan data to be received
        self.wait_for_scan_data()
        
        # Use the latest scan data to determine the state and whether the episode is done
        obs_dict = self.getState()

        # Calculate the reward based on the state and whether the episode is done
        reward = self.calculate_reward(action, obs_dict['lidar'], obs_dict['imu'], self.previous_imu_state)

        # Update previous_state with current IMU data for use in next step
        self.previous_imu_state = obs_dict['imu']
        
        # Increment the timestep counter
        self.num_timesteps += 1
        done = self.num_timesteps >= self.max_timesteps
        if done: #maybe reset() here?
            self.num_timesteps = 0
        # Prepare the info dict and check if the state is within bounds (if necessary)
        info = {}
        # Example check, you might need to adjust based on your observation space definition
        #if not all([self.observation_space.spaces[key].contains(obs_dict[key]) for key in obs_dict]):
        #    print("Exiting: Observation out of bounds.")
        #    exit()
        #state = self.preprocess_observation(obs_dict)
        #print(obs_dict)
        return obs_dict, reward, done, False, info # false for truncated

    def getState(self):
        if self.latest_scan is None:
            # Use the last known scan if the latest scan data is not available
            scan_range = self.last_known_scan
        else:
            scan_range = self.latest_scan
            self.last_known_scan = scan_range  # Update the last known scan for future use

        # Retrieve IMU data
        imu_data = self.get_imu_data()
        # Construct the state as a dictionary to match the observation space
        state = {
            "lidar": np.array(scan_range, dtype=np.float32),
            "imu": imu_data
        }
        return state
 
    def wait_for_scan_data(self):
        # Wait for the next laser scan data to be received
        while not self.new_scan_received:
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks to receive
                                                     # new messages.
    
        # At this point, self.new_scan_received is True, meaning new data is available.
        # Reset the flag for the next wait cycle.
        self.new_scan_received = False
        # Return the latest scan data for further processing.
        return self.latest_scan # list of 360 values 


    def create_observation_space(self):
        # Define the observation space for LIDAR
        lidar_space = spaces.Box(low=np.array([self.min_range]*360), 
                                 high=np.array([self.max_range]*360), 
                                 dtype=np.float32)


        # 3D orientation as a quaternion, 3D angular velocity, and 3D linear acceleration)
        imu_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Define the observation space for the camera (assuming RGB images shape 64x64)
        #camera_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # Combine these spaces into a Dict space
        observation_space = spaces.Dict({
            "lidar": lidar_space,
            "imu": imu_space,
            #"camera": camera_space,
        })
        return observation_space


    def get_clear_path_metric(self, lidar_data):
        # Assuming lidar_data is a list of distance readings ordered by angle,
        # where the front of the vehicle corresponds to the start and end of the array.
        # This function calculates the average distance in front of the vehicle to evaluate clear path.
        
        # Determine the span for the "front" sections based on your LIDAR setup
        span = 10  # Adjust based on how many points you consider as "front"
    
        # Front section is split between the start and end of the LIDAR data array
        front_section = lidar_data[:span] + lidar_data[-span:]
        
        # Calculate the average distance in this front section as the clear path metric
        clear_path_metric = sum(front_section) / len(front_section)
        return clear_path_metric


    def calculate_stability_metric(self, imu_data):
        # Extract angular velocity and linear acceleration components
        angular_velocity_vector = imu_data[4:7]  # Slices [4], [5], [6] for angular velocity x, y, z
        linear_acceleration_vector = imu_data[7:10]  # Slices [7], [8], [9] for linear acceleration x, y, z

        # Calculate the magnitude (Euclidean norm) of angular velocity and linear acceleration
        angular_velocity = np.linalg.norm(angular_velocity_vector)
        linear_acceleration = np.linalg.norm(linear_acceleration_vector)

        # Stability is higher when both angular velocity and linear acceleration are low
        stability_metric = 1 / (1 + angular_velocity + linear_acceleration)
        return stability_metric

    def calculate_smoothness_metric(self, previous_imu_data, current_imu_data):
        # Calculate the differences in angular velocity and linear acceleration
        delta_angular_velocity = np.linalg.norm(current_imu_data[4:7] - previous_imu_data[4:7])
        delta_linear_acceleration = np.linalg.norm(current_imu_data[7:10] - previous_imu_data[7:10])
    
        # Smoothness is higher when changes in angular velocity and linear acceleration are low
        smoothness_metric = 1 / (1 + delta_angular_velocity + delta_linear_acceleration)
        return smoothness_metric

    def reward_for_movement_from_actions(self, action):
        linear_vel_action = action[0]  # Assuming this is linear velocity
        angular_vel_action = action[1]  # Assuming this is angular velocity
    
        # Simple example: Reward is proportional to the linear speed minus a smaller factor of angular speed
        movement_reward = linear_vel_action - 0.4 * abs(angular_vel_action)

        # Optionally, you can set thresholds or caps to avoid excessively rewarding very high speeds
        return movement_reward


    def calculate_reward(self, action, lidar_data, imu_data, previous_imu_state=None):
        reward = 0
        c_path_metric_multiplier = 0.04 # eg  /20
        stability_metric_multiplier = 3.0
        safe_distance_multiplier = 10.0
        movement_reward_multiplier = 1.8
        smoothness_metric_multiplier = 0.7
        
        # Encourage Forward Movement
        # Assuming `get_clear_path_metric` is a method that returns a metric indicating
        # the clarity of the path directly in front of the robot (higher is clearer)
        clear_path_metric = self.get_clear_path_metric(lidar_data)
        reward += (clear_path_metric * c_path_metric_multiplier)

        movement_reward = self.reward_for_movement_from_actions(action)
        reward +=(movement_reward * movement_reward_multiplier)
        
        # Penalize Proximity to Obstacles
        # Assuming `get_min_distance` returns the minimum distance to an obstacle
        # from the lidar data, penalize if below a certain threshold
        min_distance = min(lidar_data)
        if min_distance < self.safe_distance:
            reward -= ((self.safe_distance - min_distance)* safe_distance_multiplier)
            #print('(self.safe_distance - min_distance)', -1 *(self.safe_distance - min_distance))
            
        # Use IMU for Stability
        # Assuming `calculate_stability_metric` returns a value indicating the stability
        # of the robot's movement (higher is more stable)
        stability_metric = self.calculate_stability_metric(imu_data)
        reward += (stability_metric * stability_metric_multiplier)
        
        # Optional: Consider previous state for smoothness in movement
        if previous_imu_state is not None:
            smoothness_metric = self.calculate_smoothness_metric(previous_imu_state, imu_data)
            reward += (smoothness_metric * smoothness_metric_multiplier)
            if  self.num_timesteps % 1000 == 0:
                print('clear_path:',round(clear_path_metric*c_path_metric_multiplier,4),
                      '    smoothness_metric:',
                      round(smoothness_metric*smoothness_metric_multiplier,4), '    stability:',
                      round(stability_metric*stability_metric_multiplier,4),
                      #'    min_distance:',round(min_distance,2), '< self.safe_distance:?',
                      '  movement reward:', round(movement_reward * movement_reward_multiplier,4),
                      #self.safe_distance,
                      ' distance penility:',
                      round((self.safe_distance - min_distance)* safe_distance_multiplier, 4))
                                                  
                print('total reward:\t', round(reward,4), '\n')
        
        return reward


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
        
    def reset_simulation_service_call(self):
        client = self.create_client(Empty, 'reset_simulation')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset_simulation service not available, waiting again...')
        request = Empty.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
            self.get_logger().info("Successfully called reset_simulation")
        except Exception as e:
            self.get_logger().info("Service call failed %r" % (e,))


    def reset(self):
        self.reset_simulation_service_call()
        self.num_timesteps = 0

        # Generate a reset goal for the TurtleBot

        # Instead of resetting the simulation or robot position, 
        # just ensure that any correct environment variables are re-initialized as needed
        # This could include clearing or updating goal-related flags and counters
        self.new_scan_received = False
        
        # Wait for the latest sensor data to ensure the environment is ready
        # This could be immediately after a goal is reached or any other condition that
        # you define as requiring a 'reset'
        self.wait_for_scan_data()
        
        state = self.getState()
        
        # Return the initial state and an empty dictionary, like a traditional reset
        #but without restarting the environment
        return state, {}

    def get_action_space_values(self):
        # Define low and high arrays with two elements each: one for angular velocity,
        # one for linear velocity
        low = np.array([self.min_ang_vel, self.min_linear_vel], dtype=np.float32)
        high = np.array([self.max_ang_vel, self.max_linear_vel], dtype=np.float32)
        shape_value = 2  # Now the action space has two dimensions

        return low, high, shape_value

    def imu_callback(self, msg):
        # Update the previous_imu with the last known imu data
        self.previous_imu = self.latest_imu

        # Update the latest_imu with the new data
        self.latest_imu = msg
        self.new_imu_received = True

    def get_imu_data(self):
        if self.latest_imu is None:
            return np.zeros(9)  # Return zeros if no IMU data is available

        imu_data = np.array([
            self.latest_imu.orientation.x, # [0]
            self.latest_imu.orientation.y, # [1]
            self.latest_imu.orientation.z, # [2]
            self.latest_imu.orientation.w, # [3]
            self.latest_imu.angular_velocity.x, # [4]
            self.latest_imu.angular_velocity.y, # [5]
            self.latest_imu.angular_velocity.z, # [6]
            self.latest_imu.linear_acceleration.x, # [7]
            self.latest_imu.linear_acceleration.y, # [8]
            self.latest_imu.linear_acceleration.z  # [9]
        ])
        return imu_data

    def laser_scan_callback(self, msg):
        # Assuming msg.ranges contains the distances
        processed_scan = [min(distance, self.max_range) if not np.isinf(distance) and
                          not np.isnan(distance) else self.max_range
                          for distance in msg.ranges]
        self.latest_scan = processed_scan  # ensure latest_scan is never None when used
        self.new_scan_received = True

    def get_time_info(self):
        time_info = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
        time_info += '-' + str(self.num_timesteps)
        return time_info

    def episode_finished(self):
        pass

    def preprocess_lidar_distances(self, scan_range):
        return scan_range


    def render(self, mode=None):
        pass


    def close(self):
        self.reset()
