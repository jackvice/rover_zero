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

        # Initializations
        self.old_g_distance = 0.0
        #self.position = Point(x=-2.0, y=-0.5, z=0.0)
        self.previous_position = Point(x=-2.0, y=-0.5, z=0.0)
        self.theta = 0 # yaw in radians
        self.prev_theta = 0 # yaw in radians
        # Other initializations
        self.num_timesteps = 0
        self.lidar_distances = None
        self.linear_vel = 0.0
        self.ang_vel = 0.0
        self.goal_list = goal_list if goal_list is not None else []
        self.last_step_time = self.start_time
        
        self.position = Pose()
        self.latest_imu = None
        self.previous_imu = None  # Or a default structure that matches IMU data
        
        # Assigning class parameters to the values passed in or to their default values

        self.max_env_size = None
        self.continuous = True
        self.num_lidar_obs = 12 # toal lidar: 360
        self.observation_size = 24 # 12 lidar + 10 imu + heading + distance
        self.action_size = 2
        self.min_range = 0.00001
        self.max_range = 3.9 #lidar max range is 3.8
        self.min_ang_vel=-0.5 #-1.5,
        self.max_ang_vel=0.5 #1.5,
        self.min_linear_vel=-0.02
        self.max_linear_vel=0.05
        self.previous_imu_state = None
        self.max_timesteps = 1_500
        self.latest_scan = None
        self.new_scan_received = False

        # Initializing ROS2 publishers, subscribers, and service clients
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 5)
        self.sub_odom = self.create_subscription(Odometry, 'odom', self.getOdometry, 10)
        self.reset_client = self.create_client(Empty, 'gazebo/reset_simulation')
        self.sub_scan = self.create_subscription(LaserScan, 'scan',
                                                 self.laser_scan_callback, 10)
        self.sub_scan = self.create_subscription(Imu, 'imu', self.imu_callback, 10)
        
         # Defining the action and observation spaces based on the environment parameters
        low_A, high_A, shape_value = self.get_action_space_values()
        self.action_space = spaces.Box(low=low_A, high=high_A, shape=(shape_value,),
                                       dtype=np.float32)
 
        self.observation_space = self.create_observation_space()
        
        self.respawn_goal = Respawn()
        self.respawn_goal.setGoalList(self.goal_list)
        self.goal_x = 1.0  # Default value not 0,0
        self.goal_y = 2.0  # Default value not 0,0
        self.goalbox_distance = 0.35
        self.angle_out = 135
        self.initGoal = True
        self.get_goalbox = False
        self.timeout_reward = -50
        self.reward_goal = 200
        #self.reward_collision=-50
        self.safe_distance = 0.3 # 0.3 is 12 inches which is 6" from corner

    def preprocess_observation(self, obs_dict):
        # Assuming obs_dict is your observation dictionary with 'lidar' and 'imu' keys
        lidar_data = obs_dict['lidar']
        imu_data = obs_dict['imu']
        goal_vector = obs_dict["goal_vector"]
        # Flatten and concatenate the lidar and imu data into a single vector
        flat_obs = np.concatenate([lidar_data.flatten(), imu_data.flatten(),
                                   goal_vector.flatten()])

        # Convert the flat observation array to float32 and add a batch dimension
        flat_obs = np.expand_dims(np.array(flat_obs, dtype=np.float32), axis=0)
    
        return flat_obs

    def step(self, action):
        truncated = False
        # Increment the timestep counter
        self.num_timesteps += 1
        if self.num_timesteps >= self.max_timesteps:
            self.num_timesteps = 0
            obs_dict, info = self.reset()
            done = True
            print("self.num_timesteps >= self.max_timesteps:", done)
            return obs_dict, self.timeout_reward, done, False, info # false for truncated
            
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
        obs_dict, done = self.getState()

        # Calculate the reward based on the state and whether the episode is done
        reward, done = self.calculate_reward(action, obs_dict, self.previous_imu_state)
        if self.get_goalbox == True:
            self.get_goalbox = False
            _, info = self.reset()
        # Update previous_states use in next step
        self.previous_imu_state = obs_dict['imu']
        self.prev_theta = self.theta
        info = {}
        #print("####################step obs", obs_dict,'\n')
        return obs_dict, reward, done, truncated, info # false for truncated

    def _getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x,
                                         self.goal_y - self.position.y), 6)

        return goal_distance

    def noise_reduce_scan(self, scan_data):
        # Assuming scan_data is an array of 360 elements
        reduced_scan = scan_data[::360 // self.num_lidar_obs]
            # Generate 4% Gaussian noise
        noise = np.random.normal(0, 0.04, len(reduced_scan)) * reduced_scan

        # Add the noise to the reduced scan points
        noisy_scan = reduced_scan + noise

        return reduced_scan
    
    def getState(self):
        heading = self.heading
        done = False
        if self.latest_scan is None:
            # Use the last known scan if the latest scan data is not available
            scan_range = self.noise_reduce_scan(self.last_known_scan)
        else:
            scan_range = self.noise_reduce_scan(self.latest_scan)
            self.last_known_scan = scan_range  # Update the last known scan for future use
        #self.goal_list = goal_list if goal_list is not None else []
        
        # Retrieve IMU data
        imu_data = self.get_imu_data()
        # Construct the state as a dictionary to match the observation space
        state = {
            "lidar": np.array(scan_range, dtype=np.float32),
            "imu": imu_data,
            #"camera": camera_space,
            "goal_vector": np.array([self.goalbox_distance, heading],
                                    dtype=np.float32),
        }
        return state, done

    
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
        lidar_space = spaces.Box(low=np.array([self.min_range]*12), 
                                 high=np.array([self.max_range]*12), 
                                 dtype=np.float32)


        # 3D orientation as a quaternion, 3D angular velocity, and 3D linear acceleration)
        imu_space = spaces.Box(low=-20, high=20, shape=(10,), dtype=np.float32)

        # Define the observation space for the camera (assuming RGB images shape 64x64)
        #camera_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        goal_vector_space = spaces.Box(low=np.array([0, -np.pi]), high=np.array([20, np.pi]),
                                       dtype=np.float32)
        #goal_vector_space = spaces.Box(low=-np.pi, high=20, shape=(2,), dtype=np.float32)
        # Combine these spaces into a Dict space
        observation_space = spaces.Dict({
            "lidar": lidar_space,
            "imu": imu_space,
            #"camera": camera_space,
            "goal_vector": goal_vector_space
        })
        return observation_space

    def getOdometry(self, odom):
        # Extracting orientation quaternion
        self.position = odom.pose.pose.position   
        orientation_q = odom.pose.pose.orientation
        #orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        
        # Converting quaternion to Euler angles
        #_, _, yaw = euler_from_quaternion(orientation_list)
        _, _, yaw = self.euler_from_quaternion(orientation_q.x, orientation_q.y,
                                               orientation_q.z, orientation_q.w)
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - 
                                self.position.x)

        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = heading
        #print("orientation_list", orientation_list)
        #print("self.goal_y", self.goal_y,  "self.position.y", self.position.y, 'self.goal_x',
        #      self.goal_x, "self.position.x", self.position.x)
        #print("yaw", yaw, 'goal_angle', goal_angle)
        #print('heading', heading)

        # 'yaw' is your theta value in radians
        self.theta = yaw


    def calculate_stability_metric(self, imu_data):
        # Extract angular velocity and linear acceleration components
        angular_velocity_vector = imu_data[4:7]  # Slices [4], [5], [6] angular velocity x, y, z
        linear_acceleration_vector = imu_data[7:10]  # Slices [7], [8], [9] linear accel x, y, z

        # Calculate the magnitude (Euclidean norm) of angular velocity and linear acceleration
        angular_velocity = np.linalg.norm(angular_velocity_vector)
        linear_acceleration = np.linalg.norm(linear_acceleration_vector)

        # Stability is higher when both angular velocity and linear acceleration are low
        stability_metric = 1 / (1 + angular_velocity + linear_acceleration)
        return stability_metric

    def calculate_smoothness_metric(self, previous_imu_data, current_imu_data):
        # Calculate the differences in angular velocity and linear acceleration
        delta_angular_velocity = np.linalg.norm(current_imu_data[4:7] -
                                                previous_imu_data[4:7])
        delta_linear_acceleration = np.linalg.norm(current_imu_data[7:10] -
                                                   previous_imu_data[7:10])
    
        # Smoothness is higher when changes in angular velocity and linear acceleration are low
        smoothness_metric = 1 / (1 + delta_angular_velocity + delta_linear_acceleration)
        return smoothness_metric

    def calculate_heading_reward(self, scale_factor):
        # Normalize the heading to a range between 0 and 1, where 0 means directly facing the goal
        # and 1 means facing directly away from the goal.
        normalized_heading = abs(self.heading) / math.pi
        
        # Invert the normalized heading to reward lower values (closer alignment to the goal)
        reward = 1 - normalized_heading
        
        # Optionally, scale the reward to adjust its magnitude
        scaled_reward = reward * scale_factor
        
        return scaled_reward


    def calculate_reward(self, action, obs_next, previous_imu_state=None):
        reward = 0
        time_info = self.get_time_info()
        imu_data = obs_next['imu']
        goal_distance_multiplier = 150.0
        safe_distance_multiplier = 10.0
        heading_multiplier = 0.2
        stability_metric_multiplier = 1.6
        smoothness_metric_multiplier = 0.185
        #direction_change_penalty = 0.5
        current_distance = self._getGoalDistace()

        if current_distance < self.goalbox_distance:
            print(f'{time_info}: ################################################# Goal!!')
            self.get_goalbox = True
            return self.reward_goal, True
        
        min_distance = min(obs_next['lidar'])
        if min_distance < self.safe_distance:
            #return self.reward_collision, True
            reward -= ((self.safe_distance - min_distance) * safe_distance_multiplier)
            #print(f'{time_info}: Collision!! with reward', reward)
            return reward, False

        reward_goal_dis = (self.old_g_distance - current_distance) \
            * goal_distance_multiplier
        reward += reward_goal_dis

        reward_goal_dir = self.calculate_heading_reward(heading_multiplier)
        reward -= abs(reward_goal_dir)
        
        # Penalize Proximity to Obstacles
        # Assuming `get_min_distance` returns the minimum distance to an obstacle
        # from the lidar data, penalize if below a certain threshold
 
            
        # Use IMU for Stability
        # Assuming `calculate_stability_metric` returns a value indicating the stability
        # of the robot's movement (higher is more stable)
        stability_metric = self.calculate_stability_metric(imu_data)
        reward += (stability_metric * stability_metric_multiplier)
        
        # Optional: Consider previous state for smoothness in movement
        if previous_imu_state is not None:
            smoothness_metric = self.calculate_smoothness_metric(previous_imu_state,
                                                                 imu_data)
            reward += (smoothness_metric * smoothness_metric_multiplier)
        if  self.num_timesteps % 500 == 0:
            print('\n',
                  "Goal x", round(self.goal_x,2), ', y', round(self.goal_y,2),
                  ",  current pos x", round(self.position.x, 4),
                  ", y", round(self.position.y, 4),
                  #",  self.theta", round(self.theta, 4),
                  ",  goal distance", round(current_distance, 4),
                  #",  goal vector", obs_next['goal_vector'], '\n'
                  " +reward_goal_dis", round(reward_goal_dis,6), 
                  ",  -reward_goal_dir", round(reward_goal_dir,4),
                  #",  self.old_g_distance",round(self.old_g_distance,4),
                  ',  +smoothness_metric:', round(smoothness_metric*
                                                 smoothness_metric_multiplier,4),
                  ',  +stability:', round(stability_metric*
                                         stability_metric_multiplier,4),
                  ',  total reward:', round(reward,4)
                  )
        previous_imu_state = imu_data
        self.old_g_distance = current_distance        
        return reward, False


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
        # Initializations
        if np.random.rand() > 0.5:
            # Mode 1: Fixed X, Random Y
            self.goal_x = np.random.choice([-1.5, 2])
            self.goal_y = np.random.uniform(-1, 1)
        else:
            # Mode 2: Random X, Fixed Y
            self.goal_x = np.random.uniform(-1, 1)
            self.goal_y = np.random.choice([-2, 2])

        self.position = Point(x=-2.0, y=-0.5, z=0.0)
        self.previous_position = Point(x=-2.0, y=-0.5, z=0.0)
        self.theta = 0 # yaw in radians
        self.prev_theta = 0 # yaw in radians
        # Other initializations
        self.num_timesteps = 0
        self.lidar_distances = None
        self.linear_vel = 0.0
        self.ang_vel = 0.0
        # Generate a reset goal for the TurtleBot

        # Instead of resetting the simulation or robot position, 
        # just ensure that any correct environment variables are re-initialized as needed
        # This could include clearing or updating goal-related flags and counters
        self.new_scan_received = False
        
        # Wait for the latest sensor data to ensure the environment is ready
        # This could be immediately after a goal is reached or any other condition that
        # you define as requiring a 'reset'
        self.wait_for_scan_data()
        
        state, done = self.getState()
        
        # Return the initial state and an empty dictionary, like a traditional reset
        #but without restarting the environment
        #print("###################reset obs", state, '\n')
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
    
    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
            
        return roll_x, pitch_y, yaw_z  # in radians
    
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
