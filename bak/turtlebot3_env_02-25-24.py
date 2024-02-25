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
        self.goal_list = goal_list if goal_list is not None else []
        self.max_env_size = None
        self.continuous = True
        self.num_lidar = 360
        self.max_timesteps = 22000000 #
        self.observation_size = 362 # 360 lidar + heading and distance
        self.action_size = 2
        self.min_range = -3.2 #for pi  #0.01, #0.13
        self.max_range = 11 #lidar max range is 3.8
        self.min_ang_vel=-0.5 #-1.5,
        self.max_ang_vel=0.5 #1.5,
        self.min_linear_vel=-0.03
        self.max_linear_vel=0.04
        #reward shaping
        self.goalbox_distance=0.25
        self.collision_distance=0.18 #0.13,
        self.reward_goal=200
        self.reward_collision=-10
        self.jitter_penalty = 0.5
        self.goal_delta_multiplier = 100
        self.max_timesteps = 25000
        self.angular_jitter_threshold = 0.007  # lower is more likely penality
        self.linear_jitter_threshold = 2  # lower is more likely penality
        
        self.angle_out=135
        self.goal_x = 1.0  # Default value not 0,0
        self.goal_y = 2.0  # Default value not 0,0
        self.latest_scan = None
        self.new_scan_received = False
        # Define thresholds for "jitteriness"
        self.heading = 0.0

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
        #print('init self.goal_list', self.goal_list)
        #exit()
        self.respawn_goal.setGoalList(self.goal_list)
        self.initGoal = True
        self.get_goalbox = False
        #self.position = Pose()

        #self.last_print_time = None
        self.reward_print_counter = 0
        self.old_goal_distance =  self._getGoalDistace()
        self.staying_in_threshold = 0.016 # Threshold for for staying in place depends
        # on speed

        # Defining the action and observation spaces based on the environment parameters
        low_A, high_A, shape_value = self.get_action_space_values()
        self.action_space = spaces.Box(low=low_A, high=high_A, shape=(shape_value,),
                                       dtype=np.float32)

        #create_observation_space()
        
        low_O, high_O = self.get_observation_space_values()
        # Now define the observation space with these bounds
        self.observation_space = gym.spaces.Box(low=low_O, high=high_O, dtype=np.float32)

        
        # Other initializations
        self.num_timesteps = 0
        self.lidar_distances = None
        self.linear_vel = 0.0
        self.ang_vel = 0.0

        self.last_step_time = self.start_time
        #self.seed()
        self.previous_position = None
        self.stay_place_penalty = -0.01
        self.latest_imu = None
        self.previous_imu = None  # Or a default structure that matches IMU data


    def step(self, action):
        # Set the robot's angular velocity based on the action

        self.ang_vel = action[0]
        self.linear_vel = action[1]


        # Publish the velocity command
        vel_cmd = Twist()
        #vel_cmd.linear.x = self.const_linear_vel # discrete
        vel_cmd.linear.x = float(self.linear_vel) # continuous 
        vel_cmd.angular.z = float(self.ang_vel)
        #print("vel_cmd.linear.x", vel_cmd.linear.x, 'vel_cmd.angular.z',
        #      vel_cmd.angular.z)
        self.pub_cmd_vel.publish(vel_cmd)

        # Wait for the next laser scan data to be received
        # Assuming `self.new_scan_received` is a flag that's True when new
        # data is received
        
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
        
        # Increment the timestep counter
        self.num_timesteps += 1
        if self.num_timesteps >= self.max_timesteps:
            done = True
            print("Episode timed out.")
            state, _ = self.reset()
        else:
            done = False

        # Return the current state, reward, done flag, and an empty info dict
        if not self.observation_space.contains(state):
            print("Exiting Observation out of bounds with length:",len(state),
                  ", and values:", state)
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
        #print('getState(), ######################## len(state)', len(state))

        # Check for goal achievement
        if current_distance < self.goalbox_distance:
            print(f'Goal achieved')
            self.get_goalbox = True
            done = True
        else:
            self.get_goalbox = False
        return np.array(state, dtype=np.float32), done

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
    
        # Define the observation space for IMU (assume 3D orientation, angular velocity,
        # and linear acceleration)
        imu_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf,
                                             -np.inf, - np.inf, -np.inf, -np.inf,
                                             -np.inf]), 
                               high=np.array([np.inf, np.inf, np.inf, np.inf,
                                              np.inf, np.inf, np.inf, np.inf, np.inf]), 
                               dtype=np.float32)
    
        # Define the observation space for the camera (assuming RGB images shape 64x64)
        camera_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    
        # Define spaces for heading and distance to the goal
        heading_space = spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]),
                                   dtype=np.float32)
        distance_space = spaces.Box(low=np.array([0]), high=np.array([100]),
                                    dtype=np.float32)
    
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
        # and the heading is between -pi and pi, and distance between some min and
        # max value.
        # Let's say distance can vary from 0 to some max_distance
        # you define based on your environment.

        max_distance = 10  # maximum distance, adjust based on your environment
 
        # Update the low and high bounds to include LIDAR, heading, and distance
        low = np.array([self.min_range] * 360 + [-math.pi, 0], dtype=np.float32)
        high = np.array([self.max_range] * 360 + [math.pi, max_distance],
                        dtype=np.float32)
        return low, high


    def _getGoalDistace(self):
        """
        if self.position is not None and hasattr(self.position, 'position'):
            goal_distance = round(math.hypot(self.goal_x - self.position.position.x,
                                             self.goal_y - self.position.position.y), 2)
        else:
            goal_distance = float('inf')  # or some default value
            print('inf goal ')
            print('self.position.position.x,self.position.x',
                  #self.position.position.x,
                  self.position.x)
            exit()
        return goal_distance
        """
        """
        if hasattr(self.position, 'x'):
            print('self.position.x,', self.position.x)
        else:
            print('WARNING! no self.position.x')
        """
        goal_distance = round(math.hypot(self.goal_x - self.position.x,
                                         self.goal_y - self.position.y), 4)
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
            # This might involve setting a default value, or skipping certain
            # computations
            return
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y,
                                self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

        self.heading = heading

    def navigationReward(self) -> float:
        """
        Calculates the reward based on the robot's heading relative to the goal.
        
        The reward increases as the robot's heading aligns closer to the direct 
         path towards the goal.
        The alignment is measured in radians, where a smaller difference between 
        the robot's heading
        and the goal direction results in a higher reward.
        
        Returns:
        - reward (float): The calculated reward based on the robot's heading.
        """
        # Normalize the heading to be within [-pi, pi]
        normalized_heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        
        # Calculate the alignment score as a normalized value between 0 and 1,
        # where 1 means perfect alignment
        # and 0 means the worst alignment. This is done by converting the heading's
        # absolute value to a fraction
        # of pi (since the worst case misalignment is pi) and subtracting from 1.
        alignment_score = 1 - abs(normalized_heading) / math.pi
        
        # Optionally, apply a scaling factor to amplify the importance of alignment
        scaling_factor = 1
        reward = scaling_factor * alignment_score

        return reward


    def calculate_jitter_penalty(self):
        if self.latest_imu is None or self.previous_imu is None:
            print('begin calculate_jitter_penalty(self): no penality')
            return 0  # need both current and previous IMU data
      
        # Calculate the magnitude of angular acceleration
        angular_acceleration_magnitude = np.linalg.norm([
            self.latest_imu.angular_velocity.x - self.previous_imu.angular_velocity.x,
            self.latest_imu.angular_velocity.y - self.previous_imu.angular_velocity.y,
            self.latest_imu.angular_velocity.z - self.previous_imu.angular_velocity.z
        ])
    
        # Calculate the magnitude of linear acceleration
        linear_acceleration_magnitude = np.linalg.norm([
            self.latest_imu.linear_acceleration.x -
            self.previous_imu.linear_acceleration.x,
            self.latest_imu.linear_acceleration.y -
            self.previous_imu.linear_acceleration.y,
            self.latest_imu.linear_acceleration.z -
            self.previous_imu.linear_acceleration.z
        ])
    
        # Initialize penalty

    
        # Apply penalties based on thresholds
        if angular_acceleration_magnitude > self.angular_jitter_threshold:
            return self.jitter_penalty
            #jitter_penalty -= 0.5  # Penalty value, adjust based on experimentation
            #angular_jitter = True
        else:
            return 0.0
            #angular_jitter = False
            
        #if False: #linear_acceleration_magnitude > self.linear_jitter_threshold:
        #    penalty -= 0.25  # Penalty value, adjust based on experimentation
        #    linear_jitter = True
        #else:
        #    linear_jitter = False
            
        #current_time = time.time()
        if False: #self.last_print_time is None or (current_time -
              #self.last_print_time) > 60:
            #print(current_time, self.last_print_time)
            print("\n #################################    Angular Jitter:",
                  angular_jitter,",   Linear Jitter:", linear_jitter )
            print("Angular jitter penalty, Thresh: ", self.angular_jitter_threshold ,
                  ',  Angular accel:',
                  angular_acceleration_magnitude)
            print("Linear jitter penalty, thresh:", self.linear_jitter_threshold,
                  ',  Linear accel:', linear_acceleration_magnitude)
            self.last_print_time = current_time

        return penalty

    
    def setReward(self, state: np.ndarray, action) -> float:
        """
        Calculates the reward based on the current state of the environment 
        and the action taken. 
        This function accounts for reaching the goal and collision, where 
        reaching the goal 
        provides a positive reward,  and collision provides a negative reward without 
        terminating the episode.

        Parameters:
        - state (np.ndarray): current state of the environment, including information 
        - about the robot's position, orientation, and other relevant factors.
        - action: The action taken by the robot. The specific type and structure 
           depend on the 
        - action space of the environment but is not directly used in this function.

        Returns:
        - reward (float): The calculated reward based on the current state.
        """
        reward = 0
        collisionR = 0
        nav_reward = 0
        # Check if the robot has reached the goal

        if self.get_goalbox:
            reward = self.reward_goal
            self.get_goalbox = False  # Reset the goal status
        
            # Update the goal position for continuous learning
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True)
            print("############## GOAL ACHIEVED, New Goal at x:",self.goal_x,
                  ", y:",  self.goal_y )


            # Check for collision
        elif min(self.latest_scan) < self.collision_distance:
            collisionR = reward = reward + self.reward_collision
        else:
            nav_reward = self.navigationReward()
            #reward += nav_reward
            
        self.goal_distance = self._getGoalDistace()
        delta_distance = self.old_goal_distance - self.goal_distance
        if delta_distance > 1.0 or delta_distance < -1.0: 
            delta_distance = 0.0
        self.old_goal_distance = self.goal_distance
        distance_reward = (delta_distance * self.goal_delta_multiplier)
        reward += distance_reward
        #reward += self.calculate_jitter_penalty()
        jitter_penalty = self.calculate_jitter_penalty()
        reward += jitter_penalty
        
        ####  Print every n number of calls
        if False: #self.reward_print_counter % 100 == 0:
            print(#"\n heading", round(heading,3),
                #"\t nav R:", round(nav_reward,3),
                "  jitter Rewd:", jitter_penalty,
                #'\t deltaD:', round(delta_distance,4),
                '\t deltaD Rew:', round(distance_reward,4),
                '\t G distance', round(self.goal_distance,2),
                '\t min(scan):', round(min(self.latest_scan),2),
                '\t Bump R:', collisionR,
                '\t Final Reward:', round(reward, 4))
        self.reward_print_counter +=1
        ####  Print every n number of calls

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
        self.goal_x, self.goal_y = self.generate_reset_goal()
        print('\n\n\n    In reset(), self.goal_x', self.goal_x, ', self.goal_y',
              self.goal_y)
        # Instead of resetting the simulation or robot position, 
        # just ensure that any correct environment variables are re-initialized as needed
        # This could include clearing or updating goal-related flags and counters
        self.new_scan_received = False
        self.initGoal = False  # Assuming initGoal is used to track goal initialization
                                   #status
        
        # Wait for the latest sensor data to ensure the environment is ready
        # This could be immediately after a goal is reached or any other condition that
        # you define as requiring a 'reset'
        self.wait_for_scan_data()
        
        # Once the new goal is set and the latest scan data is available,
        # calculate the initial state
        self.goal_distance = self._getGoalDistace()
        state, _ = self.getState()
        print("self.goal_distance",self.goal_distance)
        print('reset(), #################################### len(state)', len(state))
        state = np.array(state, dtype=np.float32)
        
        # Debugging prints can be adjusted or removed as necessary
        print(f"Reset state shape: {state.shape}")
        print("Reset state type", type(state))
        print("Reset state [0]", state[0])
        print("Reset state [360], heading", state[360], "Reset state [361], distance",
              state[361])
        
        # Check that the state is within the observation space bounds
        #assert self.observation_space.contains(state), "Observation out of bounds"
        if not self.observation_space.contains(state):
            print(f"State out of bounds. State: {state}")
            print(f"Observation space low bounds: {self.observation_space.low}")
            print(f"Observation space high bounds: {self.observation_space.high}")

            print(f"obs Out of bounds. Min state value: {np.min(state)}")
            print(f"Max state value: {np.max(state)}")

            assert False, "Observation out of bounds"

        
        # Return the initial state and an empty dictionary, like a traditional reset
        #but without restarting the environment
        return state, {}


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_default_state(self):
        # Return a default state that matches the expected shape and type
        # of your environment's observation space.
        # This is just an example; adjust according to your observation space.
        default_state = [0] * self.observation_size 
        return default_state

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

    def get_env_state(self):
        return self.lidar_distances



    def render(self, mode=None):
        pass


    def close(self):
        self.reset()
