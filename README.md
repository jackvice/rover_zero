# rover_zero
ROS2 Rover Zero gazebo


# Dependancies
-pytorch
-ROS2 humble
-OpenAI gym <https://github.com/openai/gym>
-https://github.com/ITTcs/gym-turtlebot3


# Gazebo Sim
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# PPO RL agent
python ppo_turtle.py

# ROS/Gazebo files
./gym-turtlebot3/gym_turtlebot3/envs/turtlebot3_env.py
/opt/ros/humble/share/turtlebot3_gazebo/launch/turtlebot3_world.launch.py
/opt/ros/humble/share/turtlebot3_gazebo/worlds/turtlebot3_world.world