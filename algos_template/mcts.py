import numpy as np
import gymnasium
import collections
from utils.utils import init_wandb
import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import wandb
import random
import time
from typing import Tuple

class MCTSNode:
    def __init__(self, state=None, parent=None, action=None):
        """
        MCTSNode class represents a node in the MCTS tree
        Each node contains the state, the parent node, the action that led to the node, the number of visits and the value of the node        
        self.children is a dictionary, where the key is the action and the value is a list of nodes
        The list is used to store nodes with the same action but different next states
        """

        self.state = state
        self.parent = parent
        self.action = action

        self.children = {}

        self.visits = 0
        self.value = 0


    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf (it has no children)

        Returns:
            bool: True if the node is a leaf, False otherwise
        """
        return len(self.children) == 0
    
    def upper_confidence_bound(self) -> float:
        """
        Calculate the UCB value of the node
        It is used to select the best child node to explore, considering the trade-off between exploitation and exploration
        An higher value means that the node is more promising or less explored

        Returns:
            float: the UCB value of the node
        """
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def get_action_upper_confidence_bound(self, action: int) -> float:
        """
        Calculate the UCB value of the action

        Args:
            action (int): the action to search

        Returns:
            float: the UCB value of the child node with the given action
        """
        if action not in self.children:
            return float('inf')
        action_value = sum(child.value for child in self.children[action])
        action_visits = sum(child.visits for child in self.children[action])
        return action_value / action_visits + np.sqrt(2 * np.log(self.visits) / action_visits)
    
    def get_max_action_upper_confidence_bound(self, action_space: int) -> int:
        """
        Return the action with the highest UCB value

        Args:
            action_space (int): the number of actions

        Returns:
            int: the action with the highest UCB value
        """
        return max(range(action_space), key=self.get_action_upper_confidence_bound)
    
    def get_children_by_action(self, action: int) -> list:
        """
        Return the list of children nodes with the given action

        Args:
            action (int): the action to search

        Returns:
            list: the list of children nodes with the given action
        """
        return self.children[action] if action in self.children else []
    
    def has_action_next_state(self, action: int, next_state: int) -> bool:
        """
        Check if the node has a child with the next_state at the given action

        Args:
            action (int): the action to check
            next_state (int): the next state to check

        Returns:
            bool: True if the child exists, False otherwise
        """
        return action in self.children and next_state in [child.state for child in self.children[action]]
    
    def get_child_by_action_and_next_state(self, action: int, next_state: int) -> 'MCTSNode':
        """
        Return the child node with the given action and next state
        The child must be in the list of children with the given action

        Args:
            action (int): the action to search
            next_state (int): the next state to search

        Returns:
            MCTSNode: the child node with the given action and next state. None if not found
        """
        if self.has_action_next_state(action, next_state):
            for child in self.children[action]:
                if child.state == next_state:
                    return child
        return None
    
    def add_child(self, action: int, child: 'MCTSNode') -> None:
        """
        Add a child to the node

        Args:
            action (int): the action that led to the child
            child (MCTSNode): the child to add
        """
        if action not in self.children:
            self.children[action] = []
        self.children[action].append(child)


# Monte Carlo Tree Search (MCTS) algorithm
class MCTS():
    def __init__(self, params, use_wandb=False):
        if params['gym_environment'] != 'TB3':
            self.env = gymnasium.make(params['gym_environment'], render_mode=params['render_mode'])
        else:
            from utils.TB3.gym_utils.gym_unity_wrapper import UnitySafetyGym
            self.env = UnitySafetyGym(editor_run=False, env_type="windows", worker_id=int(time.time())%10000, time_scale=100, no_graphics=True, max_step=100, action_space_type='discrete')

        self.env_name = params['gym_environment']
        self.state_dim = self.env.observation_space.n
        self.action_dim = self.env.action_space.n

        self.total_episodes = params['tot_episodes']
        self.step_penality = params['parameters']['step_penality']
        self.hole_score = params['parameters']['hole_score']
        self.use_wandb = use_wandb

        self.iterations = 10


    def select_and_expand(self, node) -> Tuple[MCTSNode, float, bool, int, bool]:
        """
        Select the best child node to explore and expand the tree
        It simulates the environment until the end of the episode

        Args:
            node (MCTSNode): the root node

        Returns:
            MCTSNode: the leaf node
            float: the reward obtained
            bool: True if the episode is done, False otherwise
            int: the number of steps taken
            bool: True if the episode is successful, False otherwise
        """
        steps = 0
        success = False
        while True:
            steps += 1
            # Select the best action to explore
            action = node.get_max_action_upper_confidence_bound(self.action_dim)
            # Execute the action in the environment
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            # Check if the child node already exists in the children list of the node (given the action)
            child = node.get_child_by_action_and_next_state(action, new_state)

            # Update the reward (FrozenLake)
            # If the episode is done, the reward is kept as it is if the agent reached the goal
            # Otherwise, the reward is set to the hole score
            # The reward is also penalized by the step penality
            if done:
                if reward == 1: success = True
                else: reward = self.hole_score
            reward -= self.step_penality

            # If the child node does not exist, create a new node and add it to the children list
            if child is None:
                child = MCTSNode(new_state, node, action)
                node.add_child(action, child)
                return child, reward, done, steps, success
            
            # If the episode is done, return the child node
            if done: return child, reward, done, steps, success

            # If the episode is not done, continue the exploration
            node = child


    def simulate(self) -> Tuple[float, int, bool]:
        """
        Simulate the environment until the end of the episode
        The agent takes random actions

        Returns:
        """
        total_reward = 0
        steps = 0
        success = False
        while True:
            steps += 1
            # Explore randomly
            action = self.env.action_space.sample()
            # Execute the action in the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Update the reward (FrozenLake)
            # If the episode is done, the reward is kept as it is if the agent reached the goal
            # Otherwise, the reward is set to the hole score
            # The reward is also penalized by the step penality
            if done:
                if reward == 1: success = True
                else: reward = self.hole_score
            total_reward += reward - self.step_penality

            # If the episode is done, break the loop
            if done: break

        return total_reward, steps, success


    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        backpropagate the reward from the leaf node to the root node

        Args:
            node (MCTSNode): the leaf node
            reward (float): the reward to backpropagate

        """
        # Until the node is None
        while node:
            # Update the value and the number of visits of the node
            node.visits += 1
            node.value += reward
            # Move to the parent node (None if the node is the root)
            node = node.parent

    
    def training_loop(self, seed, args_wandb=None) -> None:
        random.seed(seed)
        np.random.seed(seed)
        
        if self.use_wandb: init_wandb(args_wandb)

        rewards_list, success_list, reward_queue, success_queue = [], [], collections.deque(maxlen=100), collections.deque(maxlen=100)
        
        # Reset the environment and create the root node
        state, info = self.env.reset(seed=seed)
        root = MCTSNode(state)

        # Run the training loop
        # For each episode
        for ep in range(self.total_episodes):
            # Reset the environment and go to the root node
            self.env.reset(seed=seed)
            # Select and expand the best child node
            child, reward, done, steps, success = self.select_and_expand(root)

            # If the episode is not done, simulate the environment
            if not done:
                reward, new_steps, simulation_success = self.simulate()
                # Update the reward and the number of steps
                steps += new_steps
                success = simulation_success

            # Backpropagate the reward
            self.backpropagate(child, reward)

            reward_queue.append(reward)
            success_queue.append(int(success))
            rewards_list.append(np.mean(reward_queue))
            success_list.append(np.mean(success_queue))
            print( f"episode {ep:4d}:  reward: {reward} (mean reward: {np.mean(reward_queue):5.2f}) success: {success:3d} (mean success: {success_list[-1]:5.2f})" )
            if self.use_wandb:
                wandb.log({'mean_reward': rewards_list[-1], 'mean_success': success_list[-1]})

        
        # Close the enviornment and return the rewards list
        self.env.close()
        wandb.finish()
        return rewards_list if not self.use_wandb else None