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
    def __init__(self,
                 goal_list=None,
                 max_env_size=None,
                 continuous=True,
                 observation_size=24,
                 action_size=5,
                 min_range=0.01, #0.13
                 max_range=3.9, #lidar max range is 3.8
                 min_ang_vel=-0.5, #-1.5,
                 max_ang_vel=0.5, #1.5,
                 min_linear_vel=-0.05,
                 max_linear_vel=0.08,
                 goalbox_distance=0.25,
                 collision_distance=0.2, #0.13,
                 reward_goal=200,
                 reward_collision=-5,
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
        self.goalbox_distance = goalbox_distance
        self.collision_distance = collision_distance
        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.angle_out = angle_out
        
        self.goal_x = 1.0  # Default value not 0,0
        self.goal_y = 2.0  # Default value not 0,0

        self.latest_scan = None
        self.new_scan_received = False


        # Initializing ROS2 publishers, subscribers, and service clients
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 5)
        self.sub_odom = self.create_subscription(Odometry, 'odom', self.getOdometry, 10)
        self.reset_client = self.create_client(Empty, 'gazebo/reset_simulation')
        self.unpause_client = self.create_client(Empty, 'gazebo/unpause_physics')
        self.pause_client = self.create_client(Empty, 'gazebo/pause_physics')
        self.sub_scan = self.create_subscription(LaserScan, 'scan',
                                                 self.laser_scan_callback, 10)
        self.sub_scan = self.create_subscription(Imu, 'imu', self.imu_callback, 10)
        
        # Additional environment setup
        self.respawn_goal = Respawn()
        print('init self.goal_list', self.goal_list)
        #exit()
        self.respawn_goal.setGoalList(self.goal_list)
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()


        # Defining the action and observation spaces based on the environment parameters
        low_A, high_A, shape_value = self.get_action_space_values()
        self.action_space = spaces.Box(low=low_A, high=high_A, shape=(shape_value,),
                                       dtype=np.float32)

        
        low_O, high_O = self.get_observation_space_values()
        # Now define the observation space with these bounds
        self.observation_space = gym.spaces.Box(low=low_O, high=high_O, dtype=np.float32)

        # Other initializations
        self.num_timesteps = 0
        self.lidar_distances = None
        self.linear_vel = 0.0
        self.ang_vel = 0.0
        self.start_time = time.time()
        self.last_step_time = self.start_time
        #self.seed()
        self.previous_position = None
        self.stay_place_penalty = -0.01 

    def step(self, action):
        # Set the robot's angular velocity based on the action
        self.ang_vel = action[0]
        self.linear_vel = action[1]


        #print(action)
        # Publish the velocity command
        vel_cmd = Twist()
        #vel_cmd.linear.x = self.const_linear_vel # discrete
        vel_cmd.linear.x = float(self.linear_vel) # continuous 
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

        # Calculate the reward based on the state and whether the episode is done
        reward = self.setReward(state, action)
        self.adjustRobotVelocity(action, reward)
        
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
        #print('heading', heading)
        state.append(current_distance)
        
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


    def create_observation_space(self):
        # Define the observation space for LIDAR
        lidar_space = spaces.Box(low=np.array([self.min_range]*360), 
                                 high=np.array([self.max_range]*360), 
                                 dtype=np.float32)
    
        # Define the observation space for IMU (assuming 3D orientation, angular velocity,
        # and linear acceleration)
        imu_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -
                                             np.inf, -np.inf, -np.inf, -np.inf]), 
                               high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                              np.inf, np.inf, np.inf]), 
                               dtype=np.float32)
    
        # Define the observation space for the camera (assuming RGB images of shape 64x64)
        camera_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    
        # Define spaces for heading and distance to the goal
        heading_space = spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float32)
        distance_space = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)
    
        # Combine these spaces into a Dict space
        self.observation_space = spaces.Dict({
            "lidar": lidar_space,
            "imu": imu_space,
            "camera": camera_space,
            "heading_to_goal": heading_space,
            "distance_to_goal": distance_space
    })

    
    def get_observation_space_values(self):
        # Assuming min_range and max_range define the bounds for LIDAR values,
        # and the heading is between -pi and pi, and distance between some min and max value.
        # Let's say distance can vary from 0 to some max_distance
        # you define based on your environment.

        max_distance = 10  # maximum distance, adjust based on your environment
 
        # Update the low and high bounds to include LIDAR, heading, and distance
        low = np.array([self.min_range] * 360 + [-math.pi, 0], dtype=np.float32)
        high = np.array([self.max_range] * 360 + [math.pi, max_distance], dtype=np.float32)

        return low, high

    def _getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x,
                                         self.goal_y - self.position.y), 2)
        current_time = time.time()
        if current_time - self.start_time >= 60:  # 60 seconds = 1 minute
            # Print your message
            #print('self.goal_x - self.position.x,', (self.goal_x - self.position.x),
            #      'self.goal_y - self.position.y,', (self.goal_x - self.position.x),
            #      "goal_distance", goal_distance)
            # Update the start time for tracking
            start_time = current_time
        return goal_distance

    def generate_reset_goal(self):
        self.goal_x = 1.0  # Default value
        self.goal_y = 2.0  # Default value
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


    def navigationReward(self, heading: float) -> float:
        """
        Calculates the reward based on the robot's heading relative to the goal.
        
        Parameters:
        - heading (float): The current heading (angle) of the robot in radians, 
        -  typically calculated as the difference between the robot's orientation 
        - and the direction to the goal.

        Returns:
        - reward (float): The calculated reward based on the robot's heading.
        """
        # Calculate a reference value based on the heading's deviation from the
        # ideal direction (towards the goal).
        # The closer the heading is to the direct path to the goal, the closer
        # the reference value is to 1.
        reference = 1 - 2 * abs(heading) / math.pi
        
        # Square the reference to emphasize alignment or misalignment with the goal direction,
        # and multiply by 5 to scale the reward.
        reward = 5 * (reference ** 2)
        
        # Penalize the reward if the reference is negative, indicating a heading
        # significantly away from the goal direction.
        if reference < 0:
            reward = -reward
        return reward
    
    def calculate_jitter_penalty(self):
        # Example: Simple penalty based on angular velocity magnitude
        angular_velocity_magnitude = np.linalg.norm([
            self.latest_imu.angular_velocity.x,
            self.latest_imu.angular_velocity.y,
            self.latest_imu.angular_velocity.z
        ])
        # Define a threshold for "jitteriness"
        jitter_threshold = 0.1  # Adjust based on experimentation
        if angular_velocity_magnitude > jitter_threshold:
            #print("jitter penalty")
            return -0.5  # Penalty value, adjust based on experimentation
        return 0
    
    def penalize_staying_in_place(self):
        """Calculates a penalty if the robot stays in place."""
        if self.previous_position is None:
            self.previous_position = self.position
            return 0  # No penalty at the very beginning

        # Calculate the distance moved since the last step
        distance_moved = math.sqrt((self.position.x - self.previous_position.x) ** 2 +
                                   (self.position.y - self.previous_position.y) ** 2)

        # Update the previous position for the next call
        self.previous_position = self.position

        # Apply a penalty if the robot has moved less than a certain threshold
        if distance_moved < 0.016:  # Threshold for for staying in place depends on speed
            #print('stay in place penelty, thresh .05,  moved', distance_moved)
            return self.stay_place_penalty
        else:
            #print('NO  stay in place penelty, thresh .05,  moved', distance_moved)
            return 0
    
    def setReward(self, state: np.ndarray, action) -> float:
        """
        Calculates the reward based on the current state of the environment and the action taken. 
        This function accounts for reaching the goal and collision, where reaching the goal 
        provides a positive reward,  and collision provides a negative reward without 
        terminating the episode.

        Parameters:
        - state (np.ndarray): The current state of the environment, including information 
        - about the robot's position, orientation, and other relevant factors.
        - action: The action taken by the robot. The specific type and structure depend on the 
        - action space of the environment but is not directly used in this function.

        Returns:
        - reward (float): The calculated reward based on the current state.
        """
        reward = 0
        
        # Check if the robot has reached the goal
        if self.get_goalbox:
            reward = self.reward_goal
            self.get_goalbox = False  # Reset the goal status
        
            # Update the goal position for continuous learning
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True)
            print("############## New Goal at x:",self.goal_x, ", y:",  self.goal_y )
            self.goal_distance = self._getGoalDistace()

            # Check for collision
        elif min(self.latest_scan) < self.collision_distance:
            reward = self.reward_collision

            # Calculate navigation reward if no specific event has occurred
        else:
            #print('state[-2]  # Assume state[-2] is the heading: ',state[-2])
            heading = state[-2]  # Assume state[-2] is the heading
            reward += self.navigationReward(heading)
           
        jitter_penalty = self.calculate_jitter_penalty()
        reward += jitter_penalty
        staying_in_place_penalty = self.penalize_staying_in_place()
        reward += staying_in_place_penalty

        return reward
    

    def adjustRobotVelocity(self, action, reward):
        """
        Adjusts the robot's velocity based on the action taken and the reward received. 
        This is a placeholder function; implement velocity adjustments as needed.

        Parameters:
        - action: The action taken by the robot.
        - reward (float): The reward received after taking the action.
        """
        # Example: Stop the robot if a collision is detected (negative reward).
        if reward == self.reward_collision:
            #vel_cmd = Twist()
            #vel_cmd.linear.x = float(self.linear_vel) # continuous 
            #vel_cmd.angular.z = float(self.ang_vel)
            #self.pub_cmd_vel.publish(vel_cmd)
            self.pub_cmd_vel.publish(Twist())  # Publish a zero-velocity Twist to stop the robot
        # Additional logic to adjust the robot's velocity based on actions and rewards
        #can be added here


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
        
        # Generate a reset goal for the TurtleBot
        self.goal_x, self.goal_y = self.generate_reset_goal()
        print('\n\n\n    In reset(), self.goal_x', self.goal_x, ', self.goal_y', self.goal_y)
        # Instead of resetting the simulation or robot position, 
        # just ensure that any necessary environment variables are re-initialized as needed
        # This could include clearing or updating goal-related flags and counters
        self.new_scan_received = False
        self.initGoal = False  # Assuming initGoal is used to track goal initialization status
        
        # Wait for the latest sensor data to ensure the environment is ready
        # This could be immediately after a goal is reached or any other condition that
        # you define as requiring a 'reset'
        self.wait_for_scan_data()
        
        # Once the new goal is set and the latest scan data is available,
        # calculate the initial state
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
        # Define low and high arrays with two elements each: one for angular velocity,
        # one for linear velocity
        low = np.array([self.min_ang_vel, self.min_linear_vel], dtype=np.float32)
        high = np.array([self.max_ang_vel, self.max_linear_vel], dtype=np.float32)
        shape_value = 2  # Now the action space has two dimensions

        return low, high, shape_value

    def imu_callback(self, msg):
        # Process IMU data here
        # For simplicity, let's just store the entire IMU message
        self.latest_imu = msg
        self.new_imu_received = True  # You might want to add a separate flag for new IMU data


    def get_imu_data(self):
        if self.latest_imu is None:
            return np.zeros(9)  # Return zeros if no IMU data is available
    
        imu_data = np.array([
            self.latest_imu.orientation.x, self.latest_imu.orientation.y,\
            self.latest_imu.orientation.z, self.latest_imu.angular_velocity.x,
            self.latest_imu.angular_velocity.y, self.latest_imu.angular_velocity.z,
            self.latest_imu.linear_acceleration.x, self.latest_imu.linear_acceleration.y,
            self.latest_imu.linear_acceleration.z
        ])
        return imu_data

        
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
