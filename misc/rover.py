import gym
gym.logger.set_level(40) # hide warnings
import time
import numpy as np
import copy
import math
import os
import psutil
import signal
import sys
from scipy.stats import skew
from gym import utils, spaces
from gym_gazebo2.utils import ut_generic, ut_launch, ut_mara, ut_math, ut_gazebo, tree_urdf, general_utils
from gym.utils import seeding
from gazebo_msgs.srv import SpawnEntity
import subprocess
import argparse
import transforms3d as tf3d

# ROS 2
import rclpy


from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing mara joint angles.
from control_msgs.msg import JointTrajectoryControllerState
# from gazebo_msgs.srv import SetEntityState, DeleteEntity
from gazebo_msgs.msg import ContactState, ModelState#, GetModelList
from std_msgs.msg import String
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from ros2pkg.api import get_prefix_path
from builtin_interfaces.msg import Duration

# Algorithm specific
from PyKDL import ChainJntToJacSolver # For KDL Jacobians

class RoverZero(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None



class MARAEnv(gym.Env):
    """
    TODO. Define the environment.
    """

    def __init__(self):
        """
        Initialize the MARA environemnt
        """
        # Manage command line args
        args = ut_generic.getArgsParserMARA().parse_args()
        self.gzclient = args.gzclient
        self.realSpeed = args.realSpeed
        # self.realSpeed = True
        self.velocity = args.velocity
        self.multiInstance = args.multiInstance
        self.port = args.port

        # Set the path of the corresponding URDF file
        if self.realSpeed:
            urdf = "reinforcement_learning/mara_robot_run.urdf"
            urdfPath = get_prefix_path("mara_description") + "/share/mara_description/urdf/" + urdf
        else:
            urdf = "reinforcement_learning/mara_robot_train.urdf"
            urdfPath = get_prefix_path("mara_description") + "/share/mara_description/urdf/" + urdf

        # Launch mara in a new Process
        self.launch_subp = ut_launch.startLaunchServiceProcess(
            ut_launch.generateLaunchDescriptionMara(
                self.gzclient, self.realSpeed, self.multiInstance, self.port, urdfPath))

        # Create the node after the new ROS_DOMAIN_ID is set in generate_launch_description()
        rclpy.init()
        self.node = rclpy.create_node(self.__class__.__name__)

        # class variables
        self._observation_msg = None
        self.max_episode_steps = 1024 #default value, can be updated from baselines
        self.iterator = 0
        self.reset_jnts = True
        self._collision_msg = None

        #############################
        #   Environment hyperparams
        #############################
        # Target, where should the agent reach
        self.targetPosition = np.asarray([-0.40028, 0.095615, 0.25]) # close to the table
        # self.targetPosition =  np.asarray([round(np.random.uniform(-0.615082, -0.35426), 5), round(np.random.uniform( -0.18471, 0.1475), 5), 0.35])
        self.target_orientation = np.asarray([0., 0.7071068, 0.7071068, 0.]) # arrow looking down [w, x, y, z]
        # self.targetPosition = np.asarray([-0.386752, -0.000756, 1.40557]) # easy point
        # self.target_orientation = np.asarray([-0.4958324, 0.5041332, 0.5041331, -0.4958324]) # arrow looking opposite to MARA [w, x, y, z]

        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])

        # Initial joint position
        INITIAL_JOINTS = np.array([0., 0., 0., 0., 0., 0.])

        # # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/mara_controller/command'
        JOINT_SUBSCRIBER = '/mara_controller/state'


        # joint names:
        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'
        MOTOR4_JOINT = 'motor4'
        MOTOR5_JOINT = 'motor5'
        MOTOR6_JOINT = 'motor6'
        EE_LINK = 'ee_link'

        # Set constants for links
        WORLD = 'world'
        BASE = 'base_robot'
        MARA_MOTOR1_LINK = 'motor1_link'
        MARA_MOTOR2_LINK = 'motor2_link'
        MARA_MOTOR3_LINK = 'motor3_link'
        MARA_MOTOR4_LINK = 'motor4_link'
        MARA_MOTOR5_LINK = 'motor5_link'
        MARA_MOTOR6_LINK = 'motor6_link'
        EE_LINK = 'ee_link'

        JOINT_ORDER = [MOTOR1_JOINT,MOTOR2_JOINT, MOTOR3_JOINT,
                        MOTOR4_JOINT, MOTOR5_JOINT, MOTOR6_JOINT]
        LINK_NAMES = [ WORLD, BASE,
                        MARA_MOTOR1_LINK, MARA_MOTOR2_LINK,
                        MARA_MOTOR3_LINK, MARA_MOTOR4_LINK,
                        MARA_MOTOR5_LINK, MARA_MOTOR6_LINK, EE_LINK]

        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }
        #############################

        m_jointOrder = copy.deepcopy(JOINT_ORDER)
        m_linkNames = copy.deepcopy(LINK_NAMES)

        # Initialize target end effector position
        self.environment = {
            'jointOrder': m_jointOrder,
            'linkNames': m_linkNames,
            'reset_conditions': reset_condition,
            'tree_path': urdfPath,
            'end_effector_points': EE_POINTS,
        }

        # Subscribe to the appropriate topics, taking into account the particular robot
        self._pub = self.node.create_publisher(JointTrajectory, JOINT_PUBLISHER, qos_profile=qos_profile_sensor_data)
        self._sub = self.node.create_subscription(JointTrajectoryControllerState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._sub_coll = self.node.create_subscription(ContactState, '/gazebo_contacts', self.collision_callback, qos_profile=qos_profile_sensor_data)
        self.reset_sim = self.node.create_client(Empty, '/reset_simulation')

        # Initialize a tree structure from the robot urdf.
        #   note that the xacro of the urdf is updated by hand.
        # The urdf must be compiled.
        _, self.ur_tree = tree_urdf.treeFromFile(self.environment['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.mara_chain = self.ur_tree.getChain(self.environment['linkNames'][0], self.environment['linkNames'][-1])
        self.numJoints = self.mara_chain.getNrOfJoints()
        # Initialize a KDL Jacobian solver from the chain.
        self.jacSolver = ChainJntToJacSolver(self.mara_chain)

        self.obs_dim = self.numJoints + 6 + 12

        # # Here idially we should find the control range of the robot. Unfortunatelly in ROS/KDL there is nothing like this.
        # # I have tested this with the mujoco enviroment and the output is always same low[-1.,-1.], high[1.,1.]

        low = -np.pi * np.ones(self.numJoints)
        high = np.pi * np.ones(self.numJoints)

        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)


        spawn_cli = self.node.create_client(SpawnEntity, '/spawn_entity')

        while not spawn_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/spawn_entity service not available, waiting again...')

        modelXml = ut_gazebo.getTargetSdf()
        pose = Pose()
        pose.position.x = self.targetPosition[0]
        pose.position.y = self.targetPosition[1]
        pose.position.z = self.targetPosition[2]
        pose.orientation.x = self.target_orientation[1]
        pose.orientation.y= self.target_orientation[2]
        pose.orientation.z = self.target_orientation[3]
        pose.orientation.w = self.target_orientation[0]

        #override previous spawn_request element.
        self.spawn_request = SpawnEntity.Request()
        self.spawn_request.name = "target"
        self.spawn_request.xml = modelXml
        self.spawn_request.robot_namespace = ""
        self.spawn_request.initial_pose = pose
        self.spawn_request.reference_frame = "world"

        # #ROS2 Spawn Entity
        target_future = spawn_cli.call_async(self.spawn_request)
        rclpy.spin_until_future_complete(self.node, target_future)
        # Seed the environment
        self.seed()
        self.buffer_dist_rewards = []
        self.buffer_tot_rewards = []
        self.collided = 0

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg = message

    def collision_callback(self, message):
        """
        Callback method for the subscriber of Collision data
        """
        collision_messages = ["mara::base_robot::base_robot_collision", "ground_plane::link::collision"]
        if message.collision1_name != message.collision2_name:
            if not ((message.collision1_name in collision_messages) and (message.collision2_name in collision_messages)):
                self._collision_msg = message

    def set_episode_size(self, episode_size):
        self.max_episode_steps = episode_size

    def take_observation(self):
        """
        Take observation from the environment and return it.
        :return: state.
        """
        # # # # Take an observation
        rclpy.spin_once(self.node)
        obs_message = self._observation_msg

        # Check that the observation is not prior to the action
        # obs_message = self._observation_msg
        while obs_message is None or int(str(self._observation_msg.header.stamp.sec)+(str(self._observation_msg.header.stamp.nanosec))) < self.ros_clock:
            # print("I am in obs_message is none")
            rclpy.spin_once(self.node)
            obs_message = self._observation_msg

        # Collect the end effector points and velocities in cartesian coordinates for the processObservations state.
        # Collect the present joint angles and velocities from ROS for the state.
        lastObservations = ut_mara.processObservations(obs_message, self.environment)
        #Set observation to None after it has been read.
        self._observation_msg = None

        # Get Jacobians from present joint angles and KDL trees
        # The Jacobians consist of a 6x6 matrix getting its from from
        # (joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
        ee_link_jacobians = ut_mara.getJacobians(lastObservations, self.numJoints, self.jacSolver)
        if self.environment['linkNames'][-1] is None:
            print("End link is empty!!")
            return None
        else:
            translation, rot = general_utils.forwardKinematics(self.mara_chain,
                                                self.environment['linkNames'],
                                                lastObservations[:self.numJoints],
                                                baseLink=self.environment['linkNames'][0], # use the base_robot coordinate system
                                                endLink=self.environment['linkNames'][-1])

            current_eePos_tgt = np.ndarray.flatten(general_utils.getEePoints(self.environment['end_effector_points'], translation, rot).T)
            eePos_points = current_eePos_tgt - self.targetPosition

            eeVelocities = ut_mara.getEePointsVelocities(ee_link_jacobians, self.environment['end_effector_points'], rot, lastObservations)

            # Concatenate the information that defines the robot state
            # vector, typically denoted asrobot_id 'x'.
            state = np.r_[np.reshape(lastObservations, -1),
                          np.reshape(eePos_points, -1),
                          np.reshape(eeVelocities, -1),
                          np.reshape(current_eePos_tgt,-1),
                          np.reshape(rot.reshape(1, 9),-1)]

            return state

    def collision(self):
        # Reset if there is a collision
        if self._collision_msg is not None:
            while not self.reset_sim.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('/reset_simulation service not available, waiting again...')

            reset_future = self.reset_sim.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_future)
            self._collision_msg = None
            self.collided += 1
            return True
        else:
            return False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - action
            - observation
            - reward
            - done (status)
        """
        self.iterator+=1
        # Execute "action"
        self._pub.publish(ut_mara.getTrajectoryMessage(
            action[:self.numJoints],
            self.environment['jointOrder'],
            self.velocity))

        self.ros_clock = rclpy.clock.Clock().now().nanoseconds

        # Take an observation
        obs = self.take_observation()
        # Fetch the positions of the end-effector which are nr_dof:nr_dof+3
        rewardDist = ut_math.rmseFunc( obs[self.numJoints:(self.numJoints+3)] )

        collided = self.collision()

        reward = ut_math.computeReward(rewardDist)

        # Calculate if the env has been solved
        done = bool(self.iterator == self.max_episode_steps)

        self.buffer_dist_rewards.append(rewardDist)
        self.buffer_tot_rewards.append(reward)
        info = {}
        if self.iterator % self.max_episode_steps == 0:

            max_dist_tgt = max(self.buffer_dist_rewards)
            mean_dist_tgt = np.mean(self.buffer_dist_rewards)
            std_dist_tgt = np.std(self.buffer_dist_rewards)
            min_dist_tgt = min(self.buffer_dist_rewards)
            skew_dist_tgt = skew(self.buffer_dist_rewards)

            max_tot_rew = max(self.buffer_tot_rewards)
            mean_tot_rew = np.mean(self.buffer_tot_rewards)
            std_tot_rew = np.std(self.buffer_tot_rewards)
            min_tot_rew = min(self.buffer_tot_rewards)
            skew_tot_rew = skew(self.buffer_tot_rewards)

            num_coll = self.collided

            info = {"infos":{"ep_dist_max": max_dist_tgt,"ep_dist_mean": mean_dist_tgt,"ep_dist_min": min_dist_tgt,\
                "ep_rew_max": max_tot_rew,"ep_rew_mean": mean_tot_rew,"ep_rew_min": min_tot_rew,"num_coll": num_coll,\
                "ep_dist_skew": skew_dist_tgt,"ep_dist_std": std_dist_tgt, "ep_rew_std": std_tot_rew, "ep_rew_skew":skew_tot_rew}}
            self.buffer_dist_rewards = []
            self.buffer_tot_rewards = []
            self.collided = 0

        # Return the corresponding observations, rewards, etc.
        return obs, reward, done, info

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0

        if self.reset_jnts is True:
            # reset simulation
            while not self.reset_sim.wait_for_service(timeout_sec=1.0):
                self.node.get_logger().info('/reset_simulation service not available, waiting again...')

            reset_future = self.reset_sim.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, reset_future)

        self.ros_clock = rclpy.clock.Clock().now().nanoseconds

        # Take an observation
        obs = self.take_observation()

        # Return the corresponding observation
        return obs

    def close(self):
        print("Closing " + self.__class__.__name__ + " environment.")
        self.node.destroy_node()
        parent = psutil.Process(self.launch_subp.pid)
        for child in parent.children(recursive=True):
            child.kill()
        rclpy.shutdown()
        parent.kill()
