import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium
import collections
from utils.utils import TorchModel, init_wandb, env_success
import wandb
import random
import copy
import time
from typing import Union
	
# Deep Deterministic Policy Gradient (DDPG) algorithm
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html
# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#pseudocode

class DDPG():
    def __init__(self, params, use_wandb=False):
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'], render_mode=params['render_mode'], continuous=True)
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="windows", worker_id=int(time.time())%10000, time_scale=100, no_graphics=True, max_step=100, action_space_type="continuous")
        
        self.env_name = params['gym_environment']
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.min_action = float(self.env.action_space.low[0])
        self.max_action = float(self.env.action_space.high[0])
        
        # DNN configurations
        self.hidden_layers_actor = params['parameters']['hidden_layers_actor']     
        self.hidden_layers_critic = params['parameters']['hidden_layers_critic']
        self.nodes_hidden_layers_actor = params['parameters']['nodes_hidden_layers_actor']
        self.nodes_hidden_layers_critic = params['parameters']['nodes_hidden_layers_critic']
        self.lr_actor = params['parameters']['lr_actor_optimizer']
        self.lr_critic = params['parameters']['lr_critic_optimizer']

        # create actor model, target actor model and actor optimizer
        self.actor_net = TorchModel(self.state_dim, self.action_dim, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.target_actor_net = TorchModel(self.state_dim, self.action_dim, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_net_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)

        # init the actor weights
        # According to the chapter 7 of the DDPG paper
        for layer, name in enumerate(self.actor_net._modules):
            if "fc" in name:
                dist = 1. / np.sqrt(self.actor_net._modules[name].weight.data.size()[0])
                nn.init.uniform_(self.actor_net._modules[name].weight.data, -dist, dist)
                nn.init.uniform_(self.actor_net._modules[name].bias.data, -dist, dist)
        nn.init.uniform_(self.actor_net.output.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.actor_net.output.bias.data, -3e-3, 3e-3)

        self.update_parameters(self.actor_net, self.target_actor_net, 1.0)

        # create critic model, target critic model and critic optimizer
        self.critic_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.target_critic_net = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr_critic, weight_decay=1e-2)
        
        # init the critic weights
        # According to the chapter 7 of the DDPG paper
        for layer, name in enumerate(self.critic_net._modules):
            if "fc" in name:
                dist = 1. / np.sqrt(self.critic_net._modules[name].weight.data.size()[0])
                nn.init.uniform_(self.critic_net._modules[name].weight.data, -dist, dist)
                nn.init.uniform_(self.critic_net._modules[name].bias.data, -dist, dist)
        nn.init.uniform_(self.critic_net.output.weight.data, -3e-4, 3e-4)
        nn.init.uniform_(self.critic_net.output.bias.data, -3e-4, 3e-4)


        self.update_parameters(self.critic_net, self.target_critic_net, 1.0)

        # create actor and critic target
            
        self.gamma =  params['parameters']['gamma']
        self.tau =  params['parameters']['tau'] 

        self.noise_std = params['parameters']['noise_std']

        self.epsilon_decay = params['parameters']['eps_decay']
        self.min_exploration = params['parameters']['min_exploration']

        self.total_episodes = params['tot_episodes']
        self.memory_size = params['memory_size']
        self.batch_size = params['batch_size']
        
        self.use_wandb = use_wandb

    def get_tanh_action(self, state, target=False):
        if target:
            actions = self.target_actor_net(state)
        else:
            actions = self.actor_net(state)
        tanh_actions = F.tanh(actions)
        return tanh_actions

	
    def select_action(self, state):
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
        state = torch.Tensor(state)
        actions = self.get_tanh_action(state)
        noise = (self.noise_std ** 0.5) * torch.randn(self.action_dim)
        return actions + noise
        

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
                        action = self.select_action(
                            state
                        )
                        action = action.detach().cpu().numpy()
                        
                    action = self.env.action_space.low + (action + 1.0) * 0.5 * (self.env.action_space.high - self.env.action_space.low)
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)


                exploration_rate = max(self.min_exploration, exploration_rate * self.epsilon_decay)

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
        Update the policy using the DDPG algorithm

        Args:
            memory_buffer (list): the memory buffer containing the states, actions, rewards, next states, and dones
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

        # Update the critic model
        new_actions = self.get_tanh_action(next_states, target=True).detach()
        target_critics = self.target_critic_net(torch.cat([next_states, new_actions], dim=1)).detach()

        target = rewards + self.gamma * (1 - dones) * target_critics
        state_values = self.critic_net(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(state_values, target)

        self.critic_net_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_net_optimizer.step()

        # Update the actor model
        predicted_actions = self.get_tanh_action(states)
        critic_value = self.critic_net(torch.cat([states, predicted_actions], dim=1))
        actor_loss = -torch.mean(critic_value)

        self.actor_net_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_net_optimizer.step()

        # Update the target networks
        self.update_parameters(self.actor_net, self.target_actor_net, self.tau)
        self.update_parameters(self.critic_net, self.target_critic_net, self.tau)

            
    def update_parameters(self, source, target, tau) -> None:
        """
        Apply the soft update to the target value function
        """
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
