""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import gym_rover
import time
import numpy as np
env = gym.make('RoverZero-v0')
#env = gym.make('MARAOrient-v0')
#env = gym.make('MARACollision-v0')
#env = gym.make('MARACollisionOrient-v0')
#env = gym.make('MARACollisionOrientRandomTarget-v0')

observation, _ = env.reset()
print("########## observation ONE: #########", observation)

while True:
    print('in while')
    # take a random action
    #observation, reward, done, info = env.step(env.action_space.sample())
    observation, reward, done, _, info = env.step(np.array([-0.1,0.0]))
    print("observation", observation)
