import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
env_id = 'BipedalWalker-v3'

env = PixelObservationWrapper(gym.make(env_id, render_mode="rgb_array"))
obs = env.reset()
print('obs[0].keys())', obs[0].keys())
print('obs[pixels].shape', obs[0]['pixels'].shape)
env = PixelObservationWrapper(gym.make(env_id, render_mode="rgb_array"),
                              pixels_only=False)
print('pixels_only=False)')
obs = env.reset()
print('obs[0][state]',obs[0]['state'])
print('obs[0].keys())', obs[0].keys())
print('obs[0][pixels].shape', obs[0]['pixels'].shape)
print('obs[0][state].shape',obs[0]['state'].shape)
print('observation space shape:', env.observation_space.shape)
