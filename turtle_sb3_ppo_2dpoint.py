import gymnasium as gym
import gym_turtlebot3
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

BIPED = False

class CustomDictFeatureExtractor(nn.Module):
    def __init__(self, observation_space, **kwargs):
        super().__init__()
        # Existing initialization for 'lidar' and 'imu' processing
        
        # Update features_dim to include goal_vector's dimensions
        lidar_dim = observation_space.spaces['lidar'].shape[0]
        imu_dim = observation_space.spaces['imu'].shape[0]
        goal_vector_dim = observation_space.spaces['goal_vector'].shape[0]  # Assuming this is defined in your observation space
        
        self.features_dim = lidar_dim + imu_dim + goal_vector_dim

    def forward(self, observations: dict):
        lidar = observations['lidar']
        imu = observations['imu']
        goal_vector = observations['goal_vector']  # Extract goal_vector from observations
        
        # Concatenate all parts of the observation
        processed_observations = torch.cat([lidar, imu, goal_vector], dim=-1)
        return processed_observations


class CustomTurtleBotPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule,
                         features_extractor_class=CustomDictFeatureExtractor,
                         features_extractor_kwargs={},  # Only include if additional args needed
                         **kwargs)
        # Initialize custom actor and critic here based on the observation space
        self.actor = CustomActor(observation_space, action_space)
        self.critic = CustomCritic(observation_space)


class CustomActor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomActor, self).__init__()
        # Assuming lidar data is a flat vector of size 12 and IMU data of size 10
        self.lidar_branch = nn.Sequential(
            nn.Linear(12, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.imu_branch = nn.Sequential(
            nn.Linear(10, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        # Combine and decide
        self.combined_fc = nn.Sequential(
            nn.Linear(256, 128),  # Combined size of lidar and IMU branches
            nn.ReLU(),
            nn.Linear(128, action_space.shape[0])  # Assuming continuous actions
        )
        
    def forward(self, observation):
        # Assuming observation is a dict with keys 'lidar' and 'imu'
        lidar_out = self.lidar_branch(observation['lidar'])
        imu_out = self.imu_branch(observation['imu'])
        combined = torch.cat((lidar_out, imu_out), dim=1)
        return self.combined_fc(combined)

class CustomCritic(nn.Module):
    def __init__(self, observation_space):
        super(CustomCritic, self).__init__()
        # Branches similar to the actor
        self.lidar_branch = nn.Sequential(
            nn.Linear(12, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.imu_branch = nn.Sequential(
            nn.Linear(10, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        # Output a single value
        self.value_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, observation):
        lidar_out = self.lidar_branch(observation['lidar'])
        imu_out = self.imu_branch(observation['imu'])
        combined = torch.cat((lidar_out, imu_out), dim=1)
        return self.value_fc(combined)

    
def main():
    rclpy.init()
    total_iterations = 500
    load_saved_model = False  # Set to True if you want to load a saved model

    # Fixed TensorBoard log directory based on model/environment name
    tensorboard_log_dir = "./sb3_runs/TurtleBot3"  # Example for TurtleBot3

    if BIPED:
        load_model_path = "./models/BipedalWalker-v3_final.zip"
        env_name = 'BipedalWalker-v3'
        input_type = "MlpPolicy"
        model_name = 'biped_model'
        env = gym.make(env_name)
        env = DummyVecEnv([lambda: env])
    else:  # TurtleBot3
        load_model_path = "./models/unknown.zip"
        env_name = 'TurtleBot3_Circuit_Simple_Continuous-v0'
        #input_type = "MultiInputPolicy"
        input_type = CustomTurtleBotPolicy
        model_name = 'turtle_model'
        env = gym.make(env_name)

    if load_saved_model:
        model = PPO.load(load_model_path, env=env, tensorboard_log=tensorboard_log_dir)  # Specify tensorboard_log here
        #model.learning_rate = 2.5e-4
    else:
        hyperparams = {
            "policy": input_type,
            "env": env,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
            "tensorboard_log": tensorboard_log_dir  # Use fixed directory
        }

        model = PPO(**hyperparams)

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/',
                                             name_prefix=model_name)
    total_timesteps = total_iterations * int(1e4)
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
    print("Saving final model")
    model.save(f"./models/{env_name}_final")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
