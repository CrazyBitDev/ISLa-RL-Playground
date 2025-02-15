import gymnasium as gym
import numpy as np

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class UnitySafetyGym(gym.Env):
	
	"""
	Wrapper between the Unity Engine environment of and a custom Gym environment:

		1) Fix the sate:
			Unity scans have a size of 2 * number_of_scan (i.e., 2 values for each scan): 
			(i) distance from the first obstacle in [0, 1], 
			(ii) flag integer that detects hit/non hit in [0, 1].

		2) Reward:
			Unity returns a sparse reward signal for colliding (-1) or reaching the target (1), here we compute the dense part based on the distance.
	"""

	def __init__(self, editor_run=False, env_type="macos", worker_id=0, time_scale=100, seed=0, no_graphics=True, max_step=300, penalty_value=1, action_space_type='discrete', cost_penalty=False, lidar_precision=0.04):

		"""
		Constructor of the class.

		Args:
			render (bool):
				flag to run the environment in rendered mode, currently unused
		"""

		self.scan_number = 9 #total lidar + all additional values

		# How many additional data we are passing on the obs space and we want to remove (only cost, i.e., 1)
		self.additiona_obs = 1

		if env_type == 'macos':
			#env_path = "../macos/TB3.app"
			env_path = "utils/TB3/macos/TB3.app"
		elif env_type == 'linux':
			env_path = "utils/TB3/linux/TB3.x86_64"
		elif env_type == 'windows':
			env_path = "utils/TB3/windows/TB3/Turtlebot3.exe"
		else: raise Exception("Only macos, linux and windows are supported")

		conf_ch = EngineConfigurationChannel()
		unity_env = UnityEnvironment(env_path, worker_id=worker_id, no_graphics=no_graphics, seed=seed, side_channels=[conf_ch])
		self.env = UnityToGymWrapper(unity_env, flatten_branched=True)
		conf_ch.set_configuration_parameters(time_scale=time_scale)

		# Initialize the counter for the maximum step counter
		self.ep_step = 0
		self.max_step = max_step

		# Initialize the done flag of the environment
		self.done = False

		# Wheter to consider collisions as unsafe behaviors (i.e., they don't influence the reward)
		self.cost_penalty = cost_penalty
		self.penalty_value = penalty_value

		self.action_space_type = action_space_type

		# Override the action space of the wrapper. This is to get gym.vector.AsyncVectorEnv working
		# override action space based on user decision
		if self.action_space_type == 'continuous':
			self.action_space = gym.spaces.Box(low=np.array([-45,0]), high=np.array([45,0.05]),dtype=np.float32)
		else:
			self.action_space = gym.spaces.Discrete(6)

		# According to the previous line, we override the observation space
		# lidar_scans in [0, 1], distance [0, 1], heading [-1, 1]
		self.observation_space = gym.spaces.Box(
			np.array([0 for _ in range(self.scan_number-2)] + [0, -1]), 
			np.array([1 for _ in range(self.scan_number-2)] + [1, 1]),
			dtype=np.float64
		)

		self.lidar_precision = lidar_precision
		
	def reset(self):

		"""
		Override of the reset function of OpenAI Gym

		Returns:
			state (list):
				a list of the observation, with scan_number + 4 elements, the first part contains the information
				about the ldiar sensor and the second angle and distance, x pos and z pos with respect to the target. All the values
				are normalized between [0, 1] for the lidars and distance, heading and x,z pos in [-1, 1]
		"""

		# Reset the episode variables (step counter, done flag)
		self.ep_step = 0
		self.done = False

		state = self.env.reset()

		# Call the function that fix the state according (i.e., fix the lidar and remove the last value cost)
		state = self._fix_state(state)
		
		# We already removed cost and done values, so the distance is at index -2
		# Store the distance for the reward function 
		self.old_tg_distance = state[-2]

		return state


	def step(self, act):

		"""
		Override of the step function of OpenAI Gym

		Args:
			action (int):
				integer that represent the action that the agent must performs

		Returns:
		"""

		if self.action_space_type != 'continuous':
			discrete_actions = [[-45, 0], [0, 0], [45, 0], [-45, 0.05], [0, 0.05], [45, 0.05]]
			action = discrete_actions[act]
		else:
			action = act

		# Call the step function of the OpenAI Gym class
		obs_state, reward, done, info = self.env.step(action)
		obs_state = self._fix_state(obs_state)
		
		info['cost'] = 0
		# if np.any(np.where(obs_state[:-7] <= self.lidar_precision)[0]):
		# 	if reward != 1:
		# 		reward = -1
			
		info = {
			'tg_reach': reward == 1,
			'ag_crash': reward == -1,
			'cost': int(reward == -1),
			'timeout': self.ep_step >= self.max_step,
		}

		# Check done due to reaching target
		done = True if info['tg_reach'] else False

		# Computing the basic step reward with the new 
		# formula: a bonus to move towards the target adding a nomralization 
		# multiplier and a small penalty for each step
		# If terminal states (done) use the default value (1 or -1).
		# if not done:
		# 	#print("Not done")
		# 	r_scaling = 5
		# 	tg_distance = obs_state[-2]
		# 	reward = r_scaling * (self.old_tg_distance - tg_distance)
		# 	self.old_tg_distance = tg_distance

		if self.cost_penalty and info['cost'] > 0:
			reward -= self.penalty_value

		self.ep_step += 1
	
		return obs_state, reward, done, info['timeout'], info
	
	def _fix_state(self, state):

		"""
		Support function to convert the observation vector from the version obtained by Unity3D to our configuration.
		The orginal version provide a 2*scan_number size for the LiDAR,
		for each direction 2 value, one with the float value and one with the flag [0, 1]. 
		In this	wrapper we remove the flag, maintaining only one value for each direction

		Parameters
		----------
			state : list
				a list of the observation original observations from the environment

		Returns
		----------
			state : list
				a list of the observation, with scan_number + 4 elements, the first part contains the information
				about the ldiar sensor and the second angle and distance, x pos and z pos with respect to the target. All the values
				are normalized between [0, 1] for the lidars and distance, heading and x,z pos in [-1, 1]
		"""

		# Compute the size of the observation array that correspond to the lidar sensor, 
		# the other portion is maintened
		scan_limit = 2 * (self.scan_number-2) # -n because we have n additional value also considering the cost
		state_lidar = [s for id, s in enumerate(state[:scan_limit]) if id % 2 == 1]
		

		# Change the order of the lidar scan to the order of the wrapper (see the class declaration for details)
		lidar_ordered_1 = [s for id, s in enumerate(reversed(state_lidar)) if id % 2 == 0 ]
		lidar_ordered_2 = [s for id, s in enumerate(state_lidar) if id % 2 == 1 ]
		lidar_ordered = lidar_ordered_1 + lidar_ordered_2


		# Concatenate the ordered lidar state with the other values of the state
		# :-1 because the last value in the observation is the cost
		state_fixed = lidar_ordered + list(state[scan_limit:-self.additiona_obs])

		return np.array(state_fixed)

	
	def close(self): 
		self.env.close()

	def render(self):	
		pass
