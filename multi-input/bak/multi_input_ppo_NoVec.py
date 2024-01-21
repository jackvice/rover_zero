# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_space = env.observation_space
        act_space = env.action_space

        # Assuming the observation space is a Box and action space is Discrete
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def initialize_game(args, env, device):
    """
    Initializes the game for a single environment by setting the global step, start time, and getting 
    the initial observation and done flag.

    Parameters:
    args (Namespace): Contains arguments like the seed.
    env (Env): The single environment to reset and start a new game.
    device (Device): The device to use for tensor operations.

    Returns:
    tuple: A tuple containing the initial observation (next_obs), initial done flag (next_done), 
           global step, and start time.
    """
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)  # Resetting the single environment
    next_obs = torch.from_numpy(next_obs).float().to(device)  # Convert to torch tensor
    # Initial done flag for a single environment
    next_done = torch.tensor(0, dtype=torch.float32, device=device)  
    return next_obs, next_done, global_step, start_time

    
def take_step(step, env, action, device, rewards, global_step, writer):
    # TRY NOT TO MODIFY: execute the game and log data.
    # Assuming action is a scalar for a single environment

    next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

    next_done = torch.tensor(np.logical_or([terminated], [truncated])[0],
                             device=device, dtype=torch.float32)
    next_obs = torch.Tensor(next_obs).to(device)
    rewards[step] = torch.tensor(reward,device=device, dtype=torch.float32)#Reward is now a single value
    # Handling episode logging for single environment
    if "episode" in info:
        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
    return rewards, next_obs, next_done


def rover_main():
    args, run_name = initialize_args()
    device, writer = setup_logging_and_seeding(args, run_name)

    # env setup - make sure to call the function returned by make_env to create the environment
    env_creator = make_env(args.env_id, 0, args.capture_video, run_name)
    env = env_creator()  # Create the environment instance

    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # ALGO Logic: Storage setup
    obs, actions, logprobs, rewards, dones, values = initialize_storage(args, env, device)
    # TRY NOT TO MODIFY: start the game
    next_obs, next_done, global_step, start_time = initialize_game(args, env, device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            #print('step', step)
            global_step += 1
            obs[step] = next_obs
            print("next_done 4:", next_done)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            rewards, next_obs, next_done = take_step(step, env, action, device,
                                                     rewards, global_step, writer)
            print("step:", step,",  action:", action,",   next_obs:",
                  next_obs, ",  next_done:", next_done)
            
            if next_done:
                # If the episode is done, reset the environment
                next_obs, _ = env.reset()
                next_done = torch.tensor(0, device=device, dtype=torch.float32)
                print("next_done is", next_done)
                #next_done = torch.Tensor([0]).to(device)
                #next_done = torch.Tensor(next_done[0]).to(device)
                print("env.reset next_done :", next_done)
                
                next_obs = torch.from_numpy(next_obs).float().to(device)

        #bootstrap value if not done 
        advantages, returns = calculate_advantages_and_returns(args, next_obs, agent, device,
                                                               dones, rewards, values, next_done)
        # flatten the batch
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = \
            flatten_batches(env, obs, logprobs, actions, advantages, returns, values)
        
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                              b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss = calc_some_losses(mb_advantages, ratio, args)

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss = clip_vloss(args, newvalue, b_returns, mb_inds, b_values)
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        log_training_metrics(writer, optimizer, global_step, v_loss, pg_loss,
                             entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var, start_time)
    env.close()
    writer.close()

    
def clip_vloss(args, newvalue, b_returns, mb_inds, b_values):
     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
     v_clipped = b_values[mb_inds] + torch.clamp(
         newvalue - b_values[mb_inds],
         -args.clip_coef,
         args.clip_coef,
     )
     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
     v_loss = 0.5 * v_loss_max.mean()
     return v_loss

     
def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def calculate_advantages_and_returns(args, next_obs, agent, device, dones, rewards, values, next_done):
    """
    Calculates advantages and returns for the current batch of data.

    Parameters:
    args (Namespace): Contains arguments like gamma and gae_lambda.
    next_obs (Tensor): Next observations from the environment.
    agent (Agent): The agent to estimate the value function.
    device (Device): The device to use for tensor operations.
    dones (Tensor): Done flags from the environment.
    rewards (Tensor): Rewards obtained from the environment.
    values (Tensor): Values estimated by the agent for the current observations.
    """
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
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * \
                nextnonterminal * lastgaelam
        returns = advantages + values
    return advantages, returns


def initialize_storage(args, env, device):
    """
    Initializes storage for observations, actions, log probabilities, rewards, dones, and values 
    for a single environment.

    Parameters:
    args (Namespace): Contains arguments like the number of steps.
    env (Env): The single environment to get observation and action space.
    device (Device): The device to use for tensor operations.

    Returns:
    tuple: A tuple containing initialized tensors for obs, actions, logprobs, rewards, dones, values.
    """
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape if isinstance(env.action_space, gym.spaces.Box) else (1,)

    obs = torch.zeros((args.num_steps,) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps,) + action_shape).to(device)
    logprobs = torch.zeros(args.num_steps).to(device)
    rewards = torch.zeros(args.num_steps).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)

    return obs, actions, logprobs, rewards, dones, values


def flatten_batches(env, obs, logprobs, actions, advantages, returns, values):
    """
    Adjusts the data for training in a single environment context.

    Parameters:
    env (Env): The single environment to get observation and action space.
    obs (Tensor): Observations from the environment.
    logprobs (Tensor): Log probabilities of the actions.
    actions (Tensor): Actions taken by the agent.
    advantages (Tensor): Advantages calculated for each step.
    returns (Tensor): Returns calculated for each step.
    values (Tensor): Values estimated by the agent.
    """
    # If the data is already structured for a single environment, 
    # these lines might just pass the data through without changes.
    b_obs = obs.reshape((-1,) + env.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + env.action_space.shape
                                if isinstance(env.action_space, gym.spaces.Box) else (-1, 1))
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values






def initialize_args():
    """
    Initializes and sets up the arguments for the training process, adapted for a single environment.

    Returns:
    Namespace: The arguments after setup.
    str: The generated run name for the training.
    """
    args = tyro.cli(Args)
    # For a single environment, batch size is typically equal to the number of steps per episode
    args.batch_size = args.num_steps
    # Minibatch size can be a divisor of batch size. Adjust as needed.
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # Number of iterations is based on the total timesteps and batch size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    return args, run_name

def calc_some_losses(mb_advantages, ratio, args):
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    return pg_loss




def setup_logging_and_seeding(args, run_name):
    """
    Sets up logging and seeding for reproducibility.

    Parameters:
    args (Namespace): The arguments after setup.
    run_name (str): The generated run name for the training.
    """
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                 for key, value in vars(args).items()])),
    )

    # Seeding for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    return device, writer

def log_training_metrics(writer, optimizer, global_step, v_loss, pg_loss, entropy_loss,
                         old_approx_kl, approx_kl, clipfracs, explained_var, start_time):
    """
    Logs various training metrics to the writer.

    Parameters:
    writer (SummaryWriter): TensorBoard writer object for logging.
    optimizer (Optimizer): The optimizer used in training.
    global_step (int): The current global step in training.
    v_loss (Tensor): Value loss tensor.
    pg_loss (Tensor): Policy gradient loss tensor.
    entropy_loss (Tensor): Entropy loss tensor.
    old_approx_kl (Tensor): Old approximate KL divergence tensor.
    approx_kl (Tensor): Approximate KL divergence tensor.
    clipfracs (array): Array of clipping fractions.
    explained_var (float): Explained variance value.
    start_time (float): The start time of the training process.
    """
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)

    sps = int(global_step / (time.time() - start_time))
    print("SPS:", sps)
    writer.add_scalar("charts/SPS", sps, global_step)



    
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
if __name__ == "__main__":
    rover_main()


def perform_environment_steps(next_obs, next_done, args, env, agent, device,
                              writer, global_step, obs, dones, values,
                              actions, logprobs, rewards):
    """
    Performs steps in the single environment, collects observations, and logs data.

    Parameters:
    args (Namespace): Contains arguments like the number of steps.
    env (Env): The single environment to interact with.
    agent (Agent): The agent that decides the actions.
    device (Device): The device to use for tensor operations.
    writer (Writer): Used for logging scalar values.
    global_step (int): The global step count.
    obs (Tensor): Observations from the environment.
    dones (Tensor): Done flags from the environment.
    values (Tensor): Values estimated by the agent.
    actions (Tensor): Actions taken by the agent.
    logprobs (Tensor): Log probabilities of the actions.
    rewards (Tensor): Rewards obtained from the environment.
    """
    for step in range(args.num_steps):
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_obs, reward, done, info = env.step(action.item())  # Assuming a single action value
        next_obs = torch.from_numpy(next_obs).float().to(device)
        next_done = torch.tensor([done], dtype=torch.float32, device=device)
        rewards[step] = torch.tensor([reward], device=device)

        log_episode_info(info, writer, global_step)
        return global_step, rewards, values, obs, next_obs, actions, logprobs




def adjust_learning_rate(args, iteration, optimizer):
    """
    Adjusts the learning rate based on the current iteration.

    Parameters:
    args (Namespace): Argument parser containing learning rate and total iterations.
    iteration (int): Current iteration number.
    optimizer (Optimizer): The optimizer whose learning rate needs to be adjusted.
    """
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

