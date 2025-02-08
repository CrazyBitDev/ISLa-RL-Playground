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
        self.actor = TorchModel(self.state_dim, self.action_dim * 2, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # create critic models (two critics)
        # the critic models will output the Q value for the state-action pair
        # so the input size will be the state size + the action size
        # The use of two Q-functions to mitigate positive bias in the policy improvement step
        # that is known to degrade performance of value based methods
        self.critic1 = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic2 = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr_critic)
        
        # create value models (two value functions, one for the target)
        self.value = TorchModel(self.state_dim, 1, self.hidden_layers_value, self.nodes_hidden_layers_value)
        self.value_target = TorchModel(self.state_dim, 1, self.hidden_layers_value, self.nodes_hidden_layers_value)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr_value)

        self.gamma =  params['parameters']['gamma']
        self.tau = params['parameters']['tau']
        
        self.total_episodes = params['tot_episodes']
        self.batch_size = params['batch_size']
        self.memory_size = params['memory_size']

        self.use_wandb = use_wandb

        # update the target value function parameters
        self.update_value_parameters()



    def select_action(self, state: Union[torch.Tensor, np.array], isReparamEnabled=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select an action from the actor model
        It uses the actor model to generate a probability distribution over the actions and samples from it

        Args:
            state (torch.Tensor | np.array): the current state
            isReparamEnabled (bool): if True, the reparameterization trick will be used to sample the action

        Returns:
            actions (torch.Tensor): the selected action
            log_probabilities (torch.Tensor): the log probabilities of the selected actio
                (see SAC paper, chapter 4.2, Equation 11)
        """
        # if state is a tensor, clone and detach it
        # otherwise, convert it to a tensor
        if isinstance(state, torch.Tensor):
            state = state.clone().detach()
        else:
            state = torch.tensor(state, dtype=torch.float)

        # get the action distribution from the actor model
        actor_result = self.actor(state)
        # split the result into the mean and the standard deviation
        mu, sigma = actor_result.split(self.action_dim, dim=1)
        sigma = torch.clamp(sigma, min=1e-6, max=1.0)
        action_pdf = torch.distributions.Normal(mu, sigma)
        
        if(isReparamEnabled):
            actions_ = action_pdf.rsample()
        else:
            actions_ = action_pdf.sample()
        
        actions_max = torch.tensor(self.env.action_space.high)
        actions = torch.tanh(actions_) * actions_max
        
        log_probabilities = action_pdf.log_prob(actions_)
        log_probabilities -= torch.log(1 - actions.pow(2) + 1e-6)
        log_probabilities = log_probabilities.sum(1, keepdim = True)
        
        return actions, log_probabilities

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
        for ep in range(self.total_episodes):
            # reset the environment and the episode reward before the episode
            ep_reward = 0
            state = self.env.reset()[0]
            success = False

            # loop through the episode
            while True:
                # select the action to perform
                action, _ = self.select_action(np.array([state]))
                action = action.detach().cpu().numpy()[0]

                # Perform the action in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # update the episode reward
                ep_reward += reward
                success = env_success(self.env_name, ep_reward)
                # check if the episode is ended
                done = terminated or truncated or success

                # Store the data in the memory buffer
                memory_buffer.append([state, action, reward, next_state, done])

                # Exit condition for the episode
                if done: break
                # Update the state to the next state
                state = next_state
            
            self.update_policy(memory_buffer)

            # Update the reward list to return
            reward_queue.append(ep_reward)
            success_queue.append(success) 
            rewards_list.append(np.mean(reward_queue))
            success_list.append(np.mean(success_queue))
            print( f"episode {ep:4d}:  reward: {int(ep_reward):3d} (mean reward: {np.mean(reward_queue):5.2f}) success: {success:3d} (mean success: {success_list[-1]:5.2f})" )
            if self.use_wandb: wandb.log({'mean_reward': rewards_list[-1], 'mean_success': success_list[-1]})
      
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
        
        # Sample a batch from the memory buffer
        batch = random.sample(memory_buffer, self.batch_size)

        # Flatten the memory buffer
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Convert the lists to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.int).unsqueeze(1)

        # Compute the values and target values
        value = self.value(states).view(-1)
        target_value = self.value_target(next_states).view(-1).clone()
        target_value[dones] = 0.0

        # Compute the Q value in order to update the value function
        new_actions, log_probs = self.select_action(states, False)
        log_probs = log_probs.view(-1)
        states_new_actions = torch.cat([states, new_actions], dim=1)
        Q1_value = self.critic1(states_new_actions)
        Q2_value = self.critic2(states_new_actions)
        Q_value = torch.min(Q1_value, Q2_value)
        Q_value = Q_value.view(-1)

        # Update the value function
        self.value_optimizer.zero_grad()
        value_target = Q_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target) # 4.2, Equation 5
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        # Recompute the Q value in order to update the critic
        new_actions, log_probs = self.select_action(states, True)
        log_probs = log_probs.view(-1)
        states_new_actions = torch.cat([states, new_actions], dim=1)
        Q1_value = self.critic1(states_new_actions)
        Q2_value = self.critic2(states_new_actions)
        Q_value = torch.min(Q1_value, Q2_value)
        Q_value = Q_value.view(-1)

        # Update the actor
        self.actor_optimizer.zero_grad()
        actor_loss = log_probs - Q_value # 4.2, Equation 12
        actor_loss = actor_loss.mean()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Compute the Q value target in order to update the critic
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        Q_value = 2 * rewards + self.gamma * target_value
        Q_value = Q_value.clone().detach()

        # Compute the Q values from the critic models
        states_actions = torch.cat([states, actions], dim=1)
        Q1_value = self.critic1(states_actions).view(-1)
        Q2_value = self.critic2(states_actions).view(-1)
        
        # Update the critics
        critic1_loss = 0.5 * F.mse_loss(Q1_value, Q_value) # 4.2, Equation 7
        critic2_loss = 0.5 * F.mse_loss(Q2_value, Q_value)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Update the target value function
        self.update_value_parameters()

    def update_value_parameters(self) -> None:
        """
        Apply the soft update to the target value function
        """
        for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

