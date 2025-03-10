import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
from gymnasium.spaces import Discrete, Box
import collections
from utils.utils import TorchModel, init_wandb, env_success, reward_shaping
import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import wandb
import random
import time
import copy
from typing import Union, Tuple

# Soft Actor-Critic (SAC) algorithm
# https://arxiv.org/pdf/1812.05905
class SAC:
    def __init__(self, params, use_wandb=False):
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'], render_mode=params['render_mode'])
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="windows", worker_id=int(time.time())%10000, time_scale=100, no_graphics=True, max_step=100, action_space_type='continuous')
        
        self.env_name = params['gym_environment']

        self.state_is_discrete = isinstance(self.env.observation_space, Discrete)
        self.state_dim = self.env.observation_space.n if self.state_is_discrete else self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # set the device to cuda if available
        # and set the default device to the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)

        # DNN configurations
        self.hidden_layers_actor = params['parameters']['hidden_layers_actor']     
        self.hidden_layers_critic = params['parameters']['hidden_layers_critic']
        self.nodes_hidden_layers_actor = params['parameters']['nodes_hidden_layers_actor']
        self.nodes_hidden_layers_critic = params['parameters']['nodes_hidden_layers_critic']
        self.lr_actor = params['parameters']['lr_actor_optimizer']
        self.lr_critic = params['parameters']['lr_critic_optimizer']
        self.lr_temperature = params['parameters']['lr_temperature_optimizer']

        # Create Actor model
        # The actor model will output the mean and the standard deviation of the action distribution
        # so the output size will be the double of the action size
        self.actor_net = TorchModel(self.state_dim, self.action_dim * 2, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_net_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)

        # Create Critic models (two Q-functions)
        # the critic models will output the Q value for the state-action pair
        # so the input size will be the state size + the action size
        # The use of two Q-functions to mitigate positive bias in the policy improvement step
        # that is known to degrade performance of value based methods
        # Two Q-functions are used also in the paper example, chapter 7.1
        self.q1_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.q2_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.target_q1_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.target_q2_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.q1_net_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr_critic)
        self.q2_net_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=self.lr_critic)

        # Initialize the target Q-functions with the same weights as the Q-functions
        self.update_parameters(self.q1_net, self.target_q1_net, 1.0)
        self.update_parameters(self.q2_net, self.target_q2_net, 1.0)

        # temperature variable
        # the entropy target is set to -action_dim as in the paper
        self.entropy_target = -self.action_dim
        self.log_temperature = torch.zeros(1, requires_grad=True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_temperature], lr=self.lr_temperature)
        
        self.action_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2.0, dtype=torch.float32)
        
        self.gamma =  params['parameters']['gamma']
        self.tau = params['parameters']['tau']
        
        self.total_episodes = params['tot_episodes']
        self.batch_size = params['batch_size']
        self.memory_size = params['memory_size']

        self.epsilon_decay = params['parameters']['epsilon_decay']
        self.min_epsilon = params['parameters']['min_epsilon']

        self.use_wandb = use_wandb



    def select_action(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select an action from the actor model
        It uses the actor model to generate a probability distribution over the actions and samples from it

        Args:
            state (torch.Tensor): the current state

        Returns:
            actions (torch.Tensor): the selected action
            log_probabilities (torch.Tensor): the log probabilities of the selected actio
        """
        
        # get the action distribution from the actor model
        actor_result = self.actor_net(state)
        # split the result into the mean and the standard deviation
        mu, std = torch.chunk(actor_result, 2, dim=-1)
        std = F.softplus(std)
        dist = torch.distributions.Normal(mu, std)
        
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        adjusted_action = torch.tanh(action) * self.action_scale + self.action_bias
        adjusted_log_prob = log_prob - torch.log(self.action_scale * (1-torch.tanh(action).pow(2)) + 1e-6)
        return adjusted_action, adjusted_log_prob

    def training_loop(self, seed: int, args_wandb=None) -> Union[list, None]:
        """
        The training loop for the SAC algorithm
        It will execute the episodes in the environment and call the update_policy method to update the policy

        Args:
            seed (int): the seed for the random number generators
            args_wandb (dict): the arguments for the wandb.init method

        Returns:
            rewards_list (list): the list of rewards for each episode
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        if self.use_wandb: init_wandb(args_wandb)

        rewards_list, success_list, reward_queue, success_queue = [], [], collections.deque(maxlen=100), collections.deque(maxlen=100)
        memory_buffer = collections.deque(maxlen=self.memory_size)

        epsilon = 1.0

        for ep in range(self.total_episodes):
            # reset the environment and the episode reward before the episode
            ep_reward = 0
            if self.env_name == "TB3":
                state = self.env.reset()
            else:
                state = self.env.reset()[0]
            success = False

            # loop through the episode
            while True:
                # select the action to perform
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _ = self.select_action(
                            torch.tensor(state).float()
                        )
                        action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                # decay the epsilon value
                epsilon = max(self.min_epsilon, epsilon * self.epsilon_decay)

                # Perform the action in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                step_data = [state, action, reward, next_state, done]
                reward_shaping(self.env_name, step_data, terminated, truncated, info)
                
                # Store the data in the memory buffer
                memory_buffer.append(step_data)
                
                self.update_policy(memory_buffer)
                
                # update the episode reward
                ep_reward += reward
                # Update the state to the next state
                state = next_state
                # Check if the environment is successful
                success = env_success(self.env_name, step_data, ep_reward, terminated, truncated, info)

                # Exit condition for the episode
                if done: break
            
            
            # Update the reward list to return
            reward_queue.append(ep_reward)
            success_queue.append(success) 
            rewards_list.append(np.mean(reward_queue))
            success_list.append(np.mean(success_queue))
            print( f"episode {ep:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f}) success: {success:3d} (mean success: {success_list[-1]:5.2f})" )
            if self.use_wandb: wandb.log({'ep_reward': ep_reward, 'mean_reward': rewards_list[-1], 'mean_success': success_list[-1]})
      
        # Close the enviornment and return the rewards list
        self.env.close()
        wandb.finish()
        return rewards_list if not self.use_wandb else None


    def update_policy(self, memory_buffer: list) -> None:
        """
        Update the policy using the memory buffer
        It will sample a batch from the memory buffer and update the policy using the SAC algorithm

        Args:
            memory_buffer (list): the memory buffer containing the data to update the policy
        """
        # Check if the memory buffer has enough data to sample a batch
        if len(memory_buffer) < self.batch_size:
            return

        # Sampling a batch from the memory buffer
        batch = random.sample(memory_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        # Convert the data to float tensors
        states      = torch.tensor(states).float()
        actions     = torch.tensor(actions).float()
        next_states = torch.tensor(next_states).float()
        rewards     = torch.tensor(rewards).unsqueeze(1).float()
        dones       = torch.tensor(np.float32(dones)).unsqueeze(1).float()

        # Update the Q-functions (critic models) weights
        with torch.no_grad():
            next_actions, next_actions_log_prob = self.select_action(next_states)
            next_states_actions = torch.cat([next_states, next_actions], dim=-1)
            next_q1_target = self.target_q1_net(next_states_actions)
            next_q2_target = self.target_q2_net(next_states_actions)
            next_min_q_target = torch.min(next_q1_target, next_q2_target)
            # Part of the Formula 6 from the SAC paper
            target = rewards + self.gamma * (1.0 - dones) * ( next_min_q_target - self.log_temperature.exp() * next_actions_log_prob )
        
        states_actions = torch.cat([states, actions], dim=-1)
        for q_net, optimizer in [(self.q1_net, self.q1_net_optimizer), (self.q2_net, self.q2_net_optimizer)]:
            q_val = q_net(states_actions)
            # Formula 6 from the SAC paper
            loss = F.mse_loss(q_val, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update policy (actor model) weights
        next_actions, next_actions_log_prob = self.select_action(states)
        entropy = - self.log_temperature.exp() * next_actions_log_prob
        states_next_actions = torch.cat([states, next_actions], dim=-1)
        q1, q2 = self.q1_net(states_next_actions), self.q2_net(states_next_actions)
        q1_q2 = torch.cat([q1, q2], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        # Formula 10 from the SAC paper in gradient descent form
        actor_loss = (- min_q - entropy).mean()
        self.actor_net_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_net_optimizer.step()
        
        # Adjust temperature
        _, next_actions_log_prob = self.select_action(states)
        # Formula 17 from the SAC paper
        ent_coef_loss = -(self.log_temperature.exp() * (next_actions_log_prob + self.entropy_target).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        
        # Update target network weights
        self.update_parameters(self.q1_net, self.target_q1_net, self.tau)
        self.update_parameters(self.q2_net, self.target_q2_net, self.tau)
        

    def update_parameters(self, source, target, tau) -> None:
        """
        Apply the soft update to the target value function

        Args:
            source (TorchModel): the source model to copy the parameters from
            target (TorchModel): the target model to copy the parameters to
            tau (float): the interpolation parameter
        """
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
