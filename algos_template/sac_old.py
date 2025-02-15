import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
from gymnasium.spaces import Discrete, Box
import collections
from utils.utils import TorchModel, init_wandb, env_success
import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import wandb
import random
import time
import copy
from typing import Union, Tuple

# Soft Actor-Critic (SAC) algorithm
# https://arxiv.org/pdf/1801.01290
class SAC:
    def __init__(self, params, use_wandb=False):
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'], render_mode=params['render_mode'], continuous=True)
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="windows", worker_id=int(time.time())%10000, time_scale=100, no_graphics=True, max_step=100, action_space_type='discrete')
        
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
        self.hidden_layers_value = params['parameters']['hidden_layers_value']
        self.nodes_hidden_layers_actor = params['parameters']['nodes_hidden_layers_actor']
        self.nodes_hidden_layers_critic = params['parameters']['nodes_hidden_layers_critic']
        self.nodes_hidden_layers_value = params['parameters']['nodes_hidden_layers_value']
        self.lr_actor = params['parameters']['lr_actor_optimizer']
        self.lr_critic = params['parameters']['lr_critic_optimizer']
        self.lr_value = params['parameters']['lr_value_optimizer']

        # create actor model
        # the actor model will output the mean and the standard deviation of the action distribution
        # so the output size will be the double of the action size
        self.actor_net = TorchModel(self.state_dim, self.action_dim * 2, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_net_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)

        # create critic models (two critics)
        # the critic models will output the Q value for the state-action pair
        # so the input size will be the state size + the action size
        # The use of two Q-functions to mitigate positive bias in the policy improvement step
        # that is known to degrade performance of value based methods
        self.q1_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.q2_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.q1_net_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr_critic)
        self.q2_net_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr_critic)

        # create value function models
        self.value_net = TorchModel(self.state_dim, 1, self.hidden_layers_value, self.nodes_hidden_layers_value)
        self.target_value_net = TorchModel(self.state_dim, 1, self.hidden_layers_value, self.nodes_hidden_layers_value)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr_value)
        
        self.update_parameters(self.value_net, self.target_value_net, 1.0)

        self.action_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2.0, dtype=torch.float32)
        
        self.gamma =  params['parameters']['gamma']
        self.tau = params['parameters']['tau']
        self.epsilon = params['parameters']['epsilon']
        
        self.total_episodes = params['tot_episodes']
        self.batch_size = params['batch_size']
        self.memory_size = params['memory_size']

        self.exploration_decay = params['parameters']['exploration_decay']
        self.min_exploration = params['parameters']['min_exploration']

        self.use_wandb = use_wandb



    def select_action(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select an action from the actor model
        It uses the actor model to generate a probability distribution over the actions and samples from it

        Args:
            state (torch.Tensor | np.array): the current state

        Returns:
            actions (torch.Tensor): the selected action
            log_probabilities (torch.Tensor): the log probabilities of the selected actio
                (see SAC paper, chapter 4.2, Equation 11)
        """
        
        state = torch.FloatTensor(state)
        # get the action distribution from the actor model
        actor_result = self.actor_net(state)
        # split the result into the mean and the standard deviation
        mu, log_std = torch.chunk(actor_result, 2, dim=-1)
        std = log_std.exp()

        # adding noise
        normal = torch.distributions.Normal(0, 1)
        z = normal.sample()
        mean_std_z = mu + std*z

        action = torch.tanh(mean_std_z)
        log_prob = torch.distributions.Normal(mu, std).log_prob(mean_std_z) - torch.log(1 - action.pow(2) + self.epsilon)

        return action, log_prob

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

        exploration_rate = 1.0

        for ep in range(self.total_episodes):
            # reset the environment and the episode reward before the episode
            ep_reward = 0
            state = self.env.reset()[0]
            state = torch.as_tensor(state, dtype=torch.float32)
            success = False

            # loop through the episode
            while True:
                # select the action to perform
                if random.random() < exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _ = self.select_action(
                            state
                        )
                        action = action.detach().cpu().numpy()
                        action = self.env.action_space.low + (action + 1.0) * 0.5 * (self.env.action_space.high - self.env.action_space.low)
                        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                exploration_rate = max(self.min_exploration, exploration_rate * self.exploration_decay)

                # Perform the action in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store the data in the memory buffer
                memory_buffer.append((
                    state,
                    action,
                    reward,
                    next_state,
                    done
                ))

                # update the episode reward
                ep_reward += reward
                # Update the state to the next state
                state = next_state
                # Check if the environment is successful
                success = env_success(self.env_name, ep_reward)

                for i in range(2):
                    self.update_policy(memory_buffer)

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

        if len(memory_buffer) < self.batch_size:
            return

            
        batch = random.sample(memory_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        states      = torch.FloatTensor(states)
        actions     = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)
        dones       = torch.FloatTensor(np.float32(dones)).unsqueeze(1)

        states_actions = torch.cat([states, actions], dim=-1)
        predicted_q1_value = self.q1_net(states_actions)
        predicted_q2_value = self.q2_net(states_actions)
        predicted_value = self.value_net(states)
        new_actions, log_prob = self.select_action(states)

        # Update the critic models
        # (Equation 8)
        target_value = self.target_value_net(next_states)
        target_q_value = rewards + self.gamma * (1 - dones) * target_value   
    
        # (Equation 7)
        q1_loss = F.mse_loss(predicted_q1_value, target_q_value.detach())
        q2_loss = F.mse_loss(predicted_q2_value, target_q_value.detach())
        
        self.q1_net_optimizer.zero_grad() 
        q1_loss.backward()
        self.q1_net_optimizer.step()

        self.q2_net_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_net_optimizer.step()

        # Update the value function model
        states_new_actions = torch.cat([states, new_actions], dim=-1)
        predicted_new_q1_value = self.q1_net(states_new_actions)
        predicted_new_q2_value = self.q2_net(states_new_actions)
        predicted_new_q_value = torch.min(predicted_new_q1_value, predicted_new_q2_value)

        # (Equation 5)
        target_value_function = predicted_new_q_value - log_prob
        value_function_loss = F.mse_loss(predicted_value, target_value_function.detach())

        self.value_net_optimizer.zero_grad()
        value_function_loss.backward()
        self.value_net_optimizer.step()

        # Update the actor model
        # (Equation 12)
        actor_loss = (log_prob - predicted_new_q_value).mean()

        self.actor_net_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_net_optimizer.step()

        # Update the target value function
        self.update_parameters(self.value_net, self.target_value_net, self.tau)

    def update_parameters(self, source, target, tau) -> None:
        """
        Apply the soft update to the target value function
        """
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
