import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
from gymnasium.spaces import Discrete, Box
import collections
from utils.utils import TorchModel, init_wandb
import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import wandb
import random
import time
from typing import Tuple, Union



def one_hot(state: int, state_space_size, int) -> np.array:
    """
    Convert a state to a one-hot encoded vector

    Args:
        state (int): the state to convert
        state_space_size (int): the size of the state space

    Returns:
        state_vector (np.array): the one-hot encoded state
    """
    state_vector = np.zeros(state_space_size)
    state_vector[state] = 1
    return state_vector



class PPO:
    def __init__(self, params, use_wandb=False):
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'], render_mode=params['render_mode'])
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="windows", worker_id=int(time.time())%10000, time_scale=100, no_graphics=True, max_step=100, action_space_type='discrete')
        
        self.env_name = params['gym_environment']

        self.state_is_discrete = isinstance(self.env.observation_space, Discrete)
        self.state_dim = self.env.observation_space.n if self.state_is_discrete else self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.hidden_layers = params['parameters']['hidden_layers']        # The number of hidden layer of the neural network
        self.nodes_hidden_layers = params['parameters']['nodes_hidden_layers']

        self.lr_opt_policy = params['parameters']['lr_optimizer_pi']
        self.policy = TorchModel(self.state_dim, self.action_dim, self.hidden_layers, self.nodes_hidden_layers, last_activation=F.softmax)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_opt_policy)

        self.lr_opt_vf = params['parameters']['lr_optimizer_vf']
        self.vf = TorchModel(self.state_dim, 1, self.hidden_layers, self.nodes_hidden_layers, last_activation=F.linear)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.lr_opt_policy)

        self.gamma =  params['parameters']['gamma']
        self.clip = params['parameters']['clip']
        self.total_episodes = params['tot_episodes']
        self.epochs = params['parameters']['epochs']

        self.use_wandb = use_wandb


    def select_action(self, state) -> Tuple[int, torch.tensor]:
        """
        Select an action by sampling from the distribution generated by the policy

        Args:
            state (np.array): the state of the environment

        Returns:
            action (int): the action to take
            log_prob (torch.tensor): the log probability of the action
        """
        # Convert state to tensor and unsqueeze to add batch dimension
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Get the probabilities of each action from the policy network
        probs = self.policy(state)
        # Sample an action from the distribution
        distr = torch.distributions.Categorical(probs)
        action = distr.sample()

        return action.item(), distr.log_prob(action)


    def training_loop(self, seed: int, args_wandb=None) -> Union[list, None]:
        """
        Train the agent on the environment

        Args:
            seed (int): the seed to use for training
            args_wandb (dict): the arguments to pass to wandb

        Returns:
            rewards_list (list): the list of rewards obtained during training
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if self.use_wandb: init_wandb(args_wandb)

        rewards_list, success_list, reward_queue, success_queue = [], [], collections.deque(maxlen=100), collections.deque(maxlen=100)
        memory_buffer = []

        # Loop through the episodes
        for ep in range(self.total_episodes):
            # Reset the environment
            state = self.env.reset()[0]
            # One-hot encode the state if it is discrete
            if self.state_is_discrete:
                state = one_hot(state, self.state_dim)
            
            # Initialize the memory buffer and episode reward
            memory_buffer = []
            ep_reward = 0
            success = 0

            # Loop through the episode
            while True:
                # Select an action from the policy
                action, log_prob = self.select_action(state)
                # Take a step in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # One-hot encode the next state if it is discrete
                if self.state_is_discrete:
                    next_state = one_hot(next_state, self.state_dim)

                # Done is True if the episode is terminated or truncated
                done = terminated or truncated
                # Add the reward to the episode reward
                success += int(truncated)

                # Append the state, action, log probability, reward, and done to the memory buffer
                memory_buffer.append([state, action, log_prob, reward, done])
                # Add the reward to the episode reward
                ep_reward += reward

                # Break if the episode is done
                if done: break
                # Otherwise, set the state to the next state and continue
                state = next_state

            # Once the episode is done, append the episode reward and success to the reward and success queues
            reward_queue.append(ep_reward)
            success_queue.append(success)
            # Calculate the mean reward and success over the last 100 episodes
            rewards_list.append(np.mean(reward_queue))
            success_list.append(np.mean(success_queue))
            print( f"episode {ep:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f}) success: {success:3d} (mean success: {success_list[-1]:5.2f})" )
            if self.use_wandb: wandb.log({'mean_reward': rewards_list[-1], 'mean_success': success_list[-1]})

            # Update the policy
            self.update_policy(memory_buffer)
        
        # Close the enviornment and return the rewards list
        self.env.close()
        if self.use_wandb: wandb.finish()
        return rewards_list if not self.use_wandb else None


    def update_policy(self, memory_buffer: list) -> None:
        """
        Update the policy and value function using the PPO algorithm

        Args:
            memory_buffer (list): the memory buffer containing the states, actions, log probabilities, rewards, and dones
        """

        # Unpack the memory buffer and convert to tensors
        states = torch.tensor(
            np.array(
                [x[0] for x in memory_buffer]
            ),
            dtype=torch.float32
        )
        actions = torch.tensor(
            [x[1] for x in memory_buffer],
            dtype=torch.int64
        ).unsqueeze(1)
        old_log_probs = torch.tensor(
            [x[2] for x in memory_buffer],
            dtype=torch.float32
        ).unsqueeze(1)
        rewards = [x[3] for x in memory_buffer]
        dones = [x[4] for x in memory_buffer]

        # Calculate the returns
        G = []
        g = 0
        # Iterate through the rewards and dones in reverse order
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # Calculate the return by adding the reward to the discounted return
            g = reward + self.gamma * g # * (1 - done)
            G.insert(0, g)
        # Convert the returns to a tensor
        G = torch.tensor(G, dtype=torch.float32).unsqueeze(1)
        G = (G - G.mean()) / G.std()

        # Update the policy and value function
        # Loop through the epochs
        for _ in range(self.epochs):
            # Get the values and log probabilities from the policy
            values = self.vf(states)
            log_probs = self.policy(states).gather(1, actions)

            # Calculate the advantages and the ratio
            advantages = G - values
            advantages = (advantages - advantages.mean()) / advantages.std()
            ratio = torch.exp(log_probs - old_log_probs)

            # Calculate the policy loss and apply the gradient
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()

            # Calculate the value function loss and apply the gradient
            value_loss = F.mse_loss(values, G)
            self.vf_optimizer.zero_grad()
            value_loss.backward()
            self.vf_optimizer.step()
