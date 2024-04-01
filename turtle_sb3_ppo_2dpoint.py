import gymnasium as gym
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

BIPED = False

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
        input_type = "MultiInputPolicy"
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
