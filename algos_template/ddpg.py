import numpy as np
import torch
import torch.nn.functional as F
import gymnasium
import collections
from utils.utils import TorchModel, init_wandb
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

        # create actor and critic
        self.actor = TorchModel(self.state_dim, self.action_dim, self.hidden_layers_actor, self.nodes_hidden_layers_actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic = TorchModel(self.state_dim + self.action_dim, 1, self.hidden_layers_critic, self.nodes_hidden_layers_critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # create actor and critic target
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
            
        self.gamma =  params['parameters']['gamma']
        self.tau =  params['parameters']['tau'] 

        self.epsilon = 1.0
        self.epsilon_decay = params['parameters']['eps_decay']

        self.update_freq = params['parameters']['update_freq'] 
        self.n_updates = params['parameters']['n_updates'] 
        self.total_episodes = params['tot_episodes']
        
        self.use_wandb = use_wandb
	

    def training_loop(self, seed: int, args_wandb=None) -> Union[list, None]:
        """
        The training loop for the DDPG algorithm
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
        memory_buffer = []
        for ep in range(self.total_episodes):

            # Reset the environment and the episode reward before the episode
            if self.env_name == "TB3":
                # we cannot seed the environment if we are using Unity
                state = self.env.reset()
            else:
                state = self.env.reset(seed=seed)[0]
            ep_reward = 0
            memory_buffer.append([])

            # Loop through the episode
            while True:
                # Select the action to perform
                if np.random.rand() < self.epsilon:
                    # Random action, sample from the action space
                    action = self.env.action_space.sample()
                else:
                    # Use the actor to select the action
                    action = self.actor(torch.tensor(state, dtype=torch.float32)).detach().numpy()
                    # Add Gaussian noise to actions for exploration
                    action = np.clip(action + np.random.normal(0, 0.1), self.min_action, self.max_action)

                # Perform the action, store the data in the memory buffer and update the reward
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # Exit condition for the episode
                # reward -= 0.01  # Penalize the agent for taking time
                done = terminated or truncated

                # Store the experience in the memory buffer
                memory_buffer[-1].append([state, action, reward, next_state, done])
                ep_reward += reward
                
                if self.env_name == "TB3":
                    success = info['tg_reach']
                else:
                    success = ep_reward > 200

                # Exit condition for the episode
                if done: break
                # Update the state for the next iteration
                state = next_state

            # Update the reward list to return
            reward_queue.append(ep_reward)
            success_queue.append(int(success))
            rewards_list.append(np.mean(reward_queue))
            success_list.append(np.mean(success_queue))
            print( f"episode {ep:4d}:  reward: {ep_reward:3.2f} (mean reward: {rewards_list[-1]:5.2f}) success: {success:3d} (mean success: {success_list[-1]:5.2f})" )
            if self.use_wandb:
                wandb.log({'mean_reward': rewards_list[-1], 'mean_success': success_list[-1]})
      
            # Update
            if ep % self.update_freq == 0:
                for _ in range(self.n_updates):
                    self.update_policy(memory_buffer)
                memory_buffer = []

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
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for ep in memory_buffer:
            for state, action, reward, next_state, done in ep:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
        # Convert the lists to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute the target Q
        # target_Q has no gradient
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = rewards + self.gamma * (1 - dones) * self.critic_target(
                torch.cat([next_states, next_actions], dim=1)
            )

        # Compute the critic Q and the target critic loss
        critic_q = self.critic(
            torch.cat([states, actions], axis=1)
        )
        critic_loss = F.mse_loss(critic_q, target_Q).mean()
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks to optimize computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        self.actor_optimizer.zero_grad()
        policy_actions = self.actor(states)
        policy_loss = self.critic(
            torch.cat([states, policy_actions], axis=1)
        )
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * target_param.data + (1.0 - self.tau) * param.data
            )

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * target_param.data + (1.0 - self.tau) * param.data
            )