import gymnasium as gym
import gym_turtlebot3
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime

def main():
    rclpy.init()
    total_iterations = 10  # Define the total number of iterations you want to run
    env_name = 'TurtleBot3_Circuit_Simple_Continuous-v0'
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    # Comprehensive Hyperparameters for SAC
    hyperparams = {
        "policy": "MultiInputPolicy",
        "env": env,
        "learning_rate": 3e-4,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (1, "episode"),
        "gradient_steps": -1,
        "action_noise": None,
        "optimize_memory_usage": False,
        "ent_coef": 'auto',
        "use_sde": False,
        "sde_sample_freq": -1,
        "target_update_interval": 1,
        "target_entropy": 'auto',
        "verbose": 1,
    }

    # Timestamp for unique TensorBoard log directory within the specified tensorboard_log path
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_log_dir = f"./sb3_runs/{env_name}_{current_time}"
    hyperparams["tensorboard_log"] = tensorboard_log_dir

    model = SAC(**hyperparams)

    # Checkpoint Callback for periodic model saving
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                             name_prefix='rl_model')

    total_timesteps = total_iterations * int(1e4)
    
    # Training with periodic checkpoint saving in a single call
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

    # Final model save
    print("Saving final model")
    model.save(f"./models/{env_name}_final")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
