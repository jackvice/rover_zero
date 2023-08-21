"""
bipedal_walker_multi.py

Description:
    - This script contains an multi-input PPO agent and associated functions to 
      fuse images and telemetry with continuous action space. 

Functions:
    - Agent class: multi-input PPO agent designed to observe 600x400 images and 
      24 values of telemetry data, producing continuous actions.
    - layer_init(layer, std, bias_const): 
        Initializes a neural network layer with orthogonal weights and a constant bias.
    
    - make_env(env_id, idx, capture_video, run_name, gamma, num_frames): 
        Creates an environment instance with specific wrappers and configurations.
    
    - main(): 
        The primary function to set up the environment, agent, and training loop.
    

Dependencies:
- PyTorch
- OpenAI gym[Box2D]
- Pillow

Author: Jack Vice
Date Created: 07/20/23
Last Modified: 08/21/23

-------------------------------------
"""
import argparse
import os
import random
import time
from distutils.util import strtobool

from PIL import Image
import torchvision.transforms as transforms


import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.spaces import Box
from gym.error import DependencyNotInstalled
from frame_pixels_stack import FrameStack
import warnings
warnings.filterwarnings("ignore")



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str,
                        #default="HalfCheetah-v4",
                        default="BipedalWalker-v3",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


"""
Agent Class:
This class defines the policy and value function neural network architecture for 
the Proximal Policy Optimization (PPO) agent. The agent processes visual input 
from an environment and outputs an action choice and the associated value 
estimate.

Notes:
    - observation is shape 3x400x601 = 3x400x600 (image) + 400x3 (telemetry) 
    - Observations are normalized by 255.0 to bring the pixel values into the 
      [0, 1] range.
    - The reshaping and permute operations ensure the observations are 
      correctly aligned for the convolutional layers.
    - Telemetry is decouple from image
"""
class Agent(nn.Module): 
    def __init__(self, envs):
        super().__init__()

        # Convolutional layers for RGB frames
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),  # Single RGB frame
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )

        # determine the exact output shape using a dummy input
        #conv_out_dim = self._get_conv_out_dim((1, 3, 400, 600))

        # Fully connected layers
        self.fc = nn.Sequential(
            layer_init(nn.Linear(210224, 512)),# conv_out_dim + 24, 512)), 
            nn.ReLU(),
        )

        self.actor_mean = layer_init(nn.Linear(512, envs.single_action_space.shape[0]), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs.single_action_space.shape[0]))
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _get_conv_out_dim(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward_network(self, x_rgb, x_telemetry):
        # Convert the tensor to an image
        #x_rgb_image = x_rgb.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)  # Change shape to [400, 600, 3]
        #print(f"Shape of x_rgb_image: {x_rgb_image.shape}")  # Keep this for now
        #pil_img = Image.fromarray(x_rgb_image.cpu().numpy())
        # Save the image
        #pil_img.save('x_rgb_image.png')
        #print (x_rgb.shape)
        #exit()
        
        #x_rgb = torch.ones(x_rgb.shape[0],3,400,600).to(self.device) / 10# x_rgb / 255.0
        
        #print (x_rgb.shape[0])
        #exit()
        x_rgb = x_rgb / 255.0
        conv_out = self.conv(x_rgb)
        combined = torch.cat([conv_out, x_telemetry], dim=1)
        return self.fc(combined)
    
    def get_value(self, x):
        # Split x into x_rgb and x_telemetry
        x_rgb = x[:, :, :, :600, :]
        x_rgb = x_rgb.squeeze(1).permute(0, 3, 1, 2)  # Gives shape [1, 3, 400, 600]
        
        x_telemetry = x[:, :, :, 600:, :3]
        x_telemetry = x_telemetry.squeeze(1).squeeze(2).reshape(1, -1)  # Gives shape [1, 400*3]
    
        return self.critic(self.forward_network(x_rgb, x_telemetry))

    
    def get_action_and_value(self, x, action=None):
        x_rgb, x_telemetry = x[:, :, :, :-1, :], x[:, :, :, -1, :]  
        x_rgb = x_rgb.squeeze(1)  # Shape [batch_size, 400, 601, 3]
        x_rgb = x_rgb.permute(0, 3, 1, 2)  # Shape [batch_size, 3, 400, 601]
        
        # Extract the telemetry data for the fully connected layers
        x_telemetry = x_telemetry.squeeze(1).squeeze(-1).reshape(x_telemetry.size(0), -1)  # Shape [batch_size, 400*3]
    
        hidden = self.forward_network(x_rgb, x_telemetry)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)



def main():
    num_frames = 1
    frame_height = 400
    frame_width = 600
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = writer_start(run_name, args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, num_frames) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    ### debug
    #next_obs, _ = envs.reset(seed=args.seed)
    #print('next_obs.keys())', next_obs.keys())
    #print('next_obs[pixels] shape:', next_obs['pixels'].shape)
    #print('next_obs[state] shape:', next_obs['state'].shape)
    #print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    #print('next_obs[pixels]', next_obs['pixels'])
    #print('next_obs[state]', next_obs['state'])
    #exit()

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    #obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    obs = torch.zeros((args.num_steps, args.num_envs) + (num_frames , frame_height, frame_width+1, 3)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)

    ######### NEW
    # Ensure shape of the pixels tensor is [1, 1, 400, 600, 3]
    pixels = torch.Tensor(next_obs['pixels']).to(device)  # Assuming its shape is [1, 400, 600, 3]
    pixels = pixels.unsqueeze(1)  # Shape becomes [1, 1, 400, 600, 3]
    
    # Convert the state tensor to the desired shape
    state = torch.Tensor(next_obs['state']).to(device)  # Shape [1, 24]
    state_tensor = state.unsqueeze(1).expand(-1, 400, -1)  # Shape [1, 400, 24]

    # Reshape state tensor and reduce its width to 1 while retaining the last dimension size 3
    state_tensor_reshaped = state_tensor.reshape(1, 400, 8, 3).mean(dim=2, keepdim=True)  # Shape becomes [1, 400, 1, 3]
    state_tensor_reshaped = state_tensor_reshaped.unsqueeze(1)  # Shape becomes [1, 1, 400, 1, 3]

    # Concatenate
    next_obs = torch.cat([pixels, state_tensor_reshaped], dim=3)  # Should now be of shape [1, 1, 400, 601, 3]
    ########## end new
    print("next_obs.shape:",next_obs.shape)    

    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            ######################### NEW
            # Ensure shape of the pixels tensor is [1, 1, 400, 600, 3]
            pixels = torch.Tensor(next_obs['pixels']).to(device)  # Assuming its shape is [1, 400, 600, 3]
            pixels = pixels.unsqueeze(1)  # Shape becomes [1, 1, 400, 600, 3]
            # Convert the state tensor to the desired shape
            state = torch.Tensor(next_obs['state']).to(device)  # Shape [1, 24]
            state_tensor = state.unsqueeze(1).expand(-1, 400, -1)  # Shape [1, 400, 24]
            # Reshape state tensor and reduce its width to 1 while retaining the last dimension size 3
            state_tensor_reshaped = state_tensor.reshape(1, 400, 8, 3).mean(dim=2, keepdim=True)  # Shape becomes [1, 400, 1, 3]
            state_tensor_reshaped = state_tensor_reshaped.unsqueeze(1)  # Shape becomes [1, 1, 400, 1, 3]
            # Concatenate
            next_obs = torch.cat([pixels, state_tensor_reshaped], dim=3)  # Should now be of shape [1, 1, 400, 601, 3]
            ####################### end new
            
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"single image, global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (num_frames , frame_height, frame_width+1, 3)) 
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                              b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) /  \
                        (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef,
                                                        1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        write_stats_to_file(optimizer, v_loss, pg_loss, entropy_loss, old_approx_kl,
                            clipfracs, approx_kl, explained_var, global_step,
                            start_time, writer)
    envs.close()

def writer_start(run_name, args):    
    writer =  SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                 for key, value in vars(args).items()])),
    )
    return writer

    
def write_stats_to_file(optimizer, v_loss, pg_loss, entropy_loss, old_approx_kl,
                            clipfracs, approx_kl, explained_var, global_step,
                            start_time, writer):
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()


    
def make_env(env_id, idx, capture_video, run_name, gamma, num_frames):
    def thunk():
        if True: #capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env =  PixelObservationWrapper(env, pixels_only=False)
        #env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        #env = FrameStack(env, num_frames)
        return env

    return thunk

class Agent_4_frames(nn.Module): 
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(12, 32, 8, stride=4)),  # Adjusted for 12 channels
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(209024, 512)),
            nn.ReLU(),
        )
        self.actor_mean = layer_init(nn.Linear(512, envs.single_action_space.shape[0]), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs.single_action_space.shape[0])) # Learnable log std
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.permute(0, 1, 4, 2, 3).reshape(-1, 12, 400, 600)
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        x = x.permute(0, 1, 4, 2, 3).reshape(-1, 12, 400, 600)
        hidden = self.network(x / 255.0)
        
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)
    
if __name__ == "__main__":
    main()
