import gymnasium as gym

# List all available environments
all_envs = gym.envs.registry.values()
multi_discrete_envs = []

# Check for environments with MultiDiscrete action space
for env_spec in all_envs:
    try:
        with gym.make(env_spec.id) as env:
            if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                multi_discrete_envs.append(env_spec.id)
    except Exception:
        # Some environments might fail to instantiate, so we'll just skip those
        pass

# Print the environments with MultiDiscrete action space
print("Environments with MultiDiscrete Action Space:")
for env_id in multi_discrete_envs:
    print(env_id)
