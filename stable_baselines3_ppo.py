import gym
import gym_turtlebot3  # Ensure this imports TurtleBot3Env correctly
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    rclpy.init()
    env_name = 'TurtleBot3_Circuit_Simple_Continuous-v0'
    env = gym.make(env_name)
    #env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", env, verbose=1)
    
    model.learn(total_timesteps=int(1e4), log_interval=10)
    print("saving model")
    model.save(env_name)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

