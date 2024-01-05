import time
from time import sleep
import os
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import cv2
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import matplotlib.pyplot as plt
from gymnasium.wrappers import PixelObservationWrapper
#from gym_custom_terrain import custom_make

class CustomCNNFeatureExtractor480(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNNFeatureExtractor480, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, features_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)

class CustomCNNFeatureExtractor240(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNNFeatureExtractor240, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, features_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)

class CustomCNNFeatureExtractor120(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CustomCNNFeatureExtractor120, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, features_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


class ResizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, new_size=(120, 120)):
        super().__init__(env)
        self.new_size = new_size
        # Update the observation space to the new size
        old_space = self.observation_space.spaces['pixels']
        self.observation_space.spaces['pixels'] = gym.spaces.Box(low=old_space.low.min(),
                                                                 high=old_space.high.max(),
                                                                 shape=(new_size[0], new_size[1], old_space.shape[2]),
                                                                 dtype=old_space.dtype)

    def observation(self, observation):
        # Resize the image observation
        resized_image = cv2.resize(observation['pixels'], self.new_size, interpolation=cv2.INTER_AREA)
        observation['pixels'] = resized_image
        return observation



    
agent_type = 'ppo'

def run():
    in_size = '240'  # '120', '240', or '480'
    start_time = time.time()
    num_episodes=100
    learn_steps=10_000
    num_test_steps=1_000
    #env = custom_make("CustomTerrainAnt-v4", "images/terrain.png")
    env = gym.make("Ant-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(env, pixels_only=False)
    env = ResizeObservationWrapper(env)

    print("env.action_space: ", env.action_space)
    print("env.observation_space", env.observation_space)

    """
    if agent_type == 'ppo':
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/" )
    elif agent_type == 'sac':
        model = SAC("CnnPolicy", train_env, verbose=1, buffer_size=100_000 )
    else:
        print('no agent')
    """

    

    if in_size == '480':
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/",
                    policy_kwargs={'features_extractor_class': CustomCNNFeatureExtractor480,
                                   'features_extractor_kwargs': {'features_dim': 64}})
    elif in_size == '240':
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/",
                    policy_kwargs={'features_extractor_class': CustomCNNFeatureExtractor240,
                                   'features_extractor_kwargs': {'features_dim': 64}})
    elif in_size == '120':
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./logs/",
                    policy_kwargs={'features_extractor_class': CustomCNNFeatureExtractor120,
                                   'features_extractor_kwargs': {'features_dim': 64}})
    else:
        raise ValueError("Invalid input size specified.")
   

    # Training loop
    
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(100):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()


    obs, _ = env.reset()
    image_data = obs['pixels']

    # Display the image using Matplotlib
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()

    try:
        env.close()
    except AttributeError:
        pass  # Ignore specific AttributeError related to rendering context


# Callback for saving models
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path='./models/',
                                         name_prefix = agent_type + '_ant')

run()
