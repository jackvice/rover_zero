import gymnasium as gym
import gym_turtlebot3
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime

BIPED = False


def main():
    rclpy.init()
    # Define the total number of iterations you want to run
    total_iterations = 400
    load_saved_model = True  # Set to True if you want to load a saved model

    if BIPED:
        load_model_path = "./models/BipedalWalker-v3_final.zip"  # Path to your saved model
        env_name = 'BipedalWalker-v3'
        input_type = "MlpPolicy"
        model_name = 'biped_model'
        env = gym.make(env_name)
        env = DummyVecEnv([lambda: env])
    else: # TurtleBot3
        load_model_path = "./models/turtle_model_1690000_steps.zip"  # Path to your saved model
        env_name = 'TurtleBot3_Circuit_Simple_Continuous-v0'
        input_type = "MultiInputPolicy"
        model_name = 'turtle_model'
        env = gym.make(env_name)

    if load_saved_model:
        model = PPO.load(load_model_path, env=env)
        # Optionally adjust hyperparameters after loading
        model.learning_rate = 2.5e-4
    else:
        # Define hyperparameters
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
        }

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard_log_dir = f"./sb3_runs/{env_name}_{current_time}"
        hyperparams["tensorboard_log"] = tensorboard_log_dir

        model = PPO(**hyperparams)
        
    # Checkpoint Callback for periodic model saving
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/',
                                             name_prefix=model_name)

    total_timesteps = total_iterations * int(1e4)
    
    # Training with periodic checkpoint saving in a single call
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

    # Final model save
    print("Saving final model")
    model.save(f"./models/{env_name}_final")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
