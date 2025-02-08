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
        self.nodes_hidden_layers_actor = params['parameters']['nodes_hidden_layers_actor']
        self.nodes_hidden_layers_critic = params['parameters']['nodes_hidden_layers_critic']
        self.lr_actor = params['parameters']['lr_actor_optimizer']
        self.lr_critic = params['parameters']['lr_critic_optimizer']

        self.lr_log_entropy_coef = params['parameters']['lr_log_entropy_coef']

        # create actor model
        # the actor model will output the mean and the standard deviation of the action distribution
        # so the output size will be the double of the action size
        self.actor = TorchModel(self.state_dim, self.action_dim * 2, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_target = TorchModel(self.state_dim, self.action_dim * 2, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # create critic models (two critics)
        # the critic models will output the Q value for the state-action pair
        # so the input size will be the state size + the action size
        # The use of two Q-functions to mitigate positive bias in the policy improvement step
        # that is known to degrade performance of value based methods
        self.critic1 = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic1_target = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2 = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic2_target = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr_critic)

        self.log_entropy_coef = torch.zeros(1, requires_grad=True)
        self.log_entropy_coef_optimizer = torch.optim.Adam([self.log_entropy_coef], lr=self.lr_log_entropy_coef)

        self.action_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2.0, dtype=torch.float32)
        
        self.gamma =  params['parameters']['gamma']
        self.tau = params['parameters']['tau']
        
        self.total_episodes = params['tot_episodes']
        self.batch_size = params['batch_size']
        self.memory_size = params['memory_size']

        self.target_entropy = params['target_entropy']

        self.use_wandb = use_wandb



    def select_action(self, state, target=False) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # get the action distribution from the actor model
        if target:
            actor_result = self.actor_target(state)
        else:
            actor_result = self.actor(state)
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
        for ep in range(self.total_episodes):
            # reset the environment and the episode reward before the episode
            ep_reward = 0
            state = self.env.reset()[0]
            state = torch.as_tensor(state, dtype=torch.float32)
            success = False

            # loop through the episode
            while True:
                # select the action to perform
                with torch.no_grad():
                    action, _ = self.select_action(
                        state
                    )
                    action = action.detach().cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

                # Perform the action in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = torch.as_tensor(next_state, dtype=torch.float32)
                # update the episode reward
                ep_reward += reward
                success = env_success(self.env_name, ep_reward)

                # Store the data in the memory buffer
                memory_buffer.append([
                    state,
                    torch.as_tensor(action),
                    torch.as_tensor(reward, dtype=torch.float32),
                    next_state,
                    torch.as_tensor(terminated, dtype=torch.float32)
                ])

                self.update_policy(memory_buffer)

                done = terminated or truncated
                # Exit condition for the episode
                if done: break
                # Update the state to the next state
                state = next_state
            
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

        ## Sample a batch from the memory buffer
        #batch = random.sample(memory_buffer, self.batch_size)

        ## Flatten the memory buffer
        #states, actions, rewards, next_states, dones = [], [], [], [], []
        #for state, action, reward, next_state, done in batch:
        #    states.append(state)
        #    actions.append(action)
        #    rewards.append(reward)
        #    next_states.append(next_state)
        #    dones.append(done)

        states, actions, rewards, next_states, dones = [torch.stack(b) for b in zip(*random.choices(memory_buffer, k=self.batch_size))]

        ## Convert the lists to tensors
        #states = torch.tensor(np.array(states), dtype=torch.float32)
        #actions = torch.tensor(np.array(actions))
        #rewards = torch.tensor(rewards, dtype=torch.float32)
        #next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        #dones = torch.tensor(dones, dtype=torch.int).unsqueeze(1)

        # Calculate the targets
        with torch.no_grad():
            next_action, next_action_log_prob = self.select_action(next_states, target=True)
            next_critic1_target = self.critic1_target(torch.cat([next_states, next_action], dim=-1))
            next_critic2_target = self.critic2_target(torch.cat([next_states, next_action], dim=-1))
            min_next_critic_target = torch.min(next_critic1_target, next_critic2_target)

            dones = dones.view(-1, 1)
            target = rewards.unsqueeze(1) + self.gamma * (1 - dones) * min_next_critic_target - self.log_entropy_coef.exp() * next_action_log_prob

        # Update the critic models
        critic1_value = self.critic1(torch.cat([states, actions], dim=-1))
        critic1_loss = F.mse_loss(critic1_value, target)
        self.critic1_optimizer.zero_grad() 
        critic1_loss.backward()
        self.critic1_optimizer.step()

        critic2_value = self.critic2(torch.cat([states, actions], dim=-1))
        critic2_loss = F.mse_loss(critic2_value, target)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update the actor model
        action, action_log_prob = self.select_action(states)
        entropy = -self.log_entropy_coef.exp() * action_log_prob
        critic1_value = self.critic1(torch.cat([states, action], dim=-1))
        critic2_value = self.critic2(torch.cat([states, action], dim=-1))
        cat_critic_values = torch.cat([critic1_value, critic2_value], dim=-1)
        min_critic_values = torch.min(cat_critic_values, 1, keepdim=True)[0]
        actor_loss = (-min_critic_values - entropy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the log entropy coefficient
        _, action_log_prob = self.select_action(states)
        entropy_coef_loss = -(self.log_entropy_coef.exp() * (action_log_prob + self.target_entropy).detach()).mean()
        self.log_entropy_coef_optimizer.zero_grad()
        entropy_coef_loss.backward()
        self.log_entropy_coef_optimizer.step()


        # Update the target value function
        self.update_parameters(self.critic1, self.critic1_target)
        self.update_parameters(self.critic2, self.critic2_target)
        self.update_parameters(self.actor, self.actor_target)

    def update_parameters(self, source, target) -> None:
        """
        Apply the soft update to the target value function
        """
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1-self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

