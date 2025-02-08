
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CYAN_COL = '\033[96m'
BLUE_COL = '\033[94m'
RED_COL = '\033[91m'
GREEN_COL = '\033[92m'
YELLOW_COL = '\033[93m'
RESET_COL = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def check_parameters(method):
    if method['name'] not in ['PPO', 'REINFORCE', 'DDPG', 'SAC', 'MCTS', 'QLEARNING']: 
        raise ValueError(f"{RED_COL}{method['name']} method is not supported! Select one of the following methods: ['PPO', 'DDPG', 'SAC', 'MCTS', 'QLEARNING'] in the yaml file.{RESET_COL}")
    

class TorchModel(nn.Module):
	"""
	Class that generates a neural network with PyTorch and specific parameters.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	"""
	
	# Initialize the neural network
	def __init__(self, nInputs, nOutputs, nLayer, nNodes, last_activation=F.linear):
		
		super(TorchModel, self).__init__()
		self.nLayer = nLayer
		self.last_activation= last_activation

		# input layer
		self.fc1 = nn.Linear(nInputs, nNodes)

		#hidden layers
		for i in range(nLayer):
			layer_name = f"fc{i+2}"
			self.add_module(layer_name, nn.Linear(nNodes, nNodes))  

		#output
		self.output = nn.Linear(nNodes, nOutputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		for i in range(2, self.nLayer + 2):
			x = F.relu(getattr(self, f'fc{i}')(x).to(x.dtype))
		x = self.output(x)
		return x if self.last_activation == F.linear else self.last_activation(x, dim=1)
	

def init_wandb(args):
    wandb.init(
        name=args['run_name'],
        project=args['project'],
        entity=args['entity'],
        mode= "online",
        save_code=False,
        config=args
    )


def plot_results(results):

	# results is a list of dictionaries where we have [{'method': 'name_method', 'mean_rew': [[r1,...,rn] (seed 0), [r1,...,rn], [r1,..., rn]], 'training_episodes': episodes}]

	# we take the training episodes from the first dictionary as if we want to compare two methods they should have performed the same number pof episodes
	t = list(range(0, results[0]['training_episodes']))

	# Plotting
	sns.set_style("darkgrid")
	plt.figure(figsize=(8, 6))  # Set the figure size
 
	for dict in results:
		data = {'Environment Step': [], 'Mean Reward': []}
		for _, rewards in enumerate(dict['mean_rewards']):
			for step, reward in zip(t, rewards):
				data['Environment Step'].append(step)
				data['Mean Reward'].append(reward)
		df = pd.DataFrame(data)
		
		sns.lineplot(data=df, x='Environment Step', y='Mean Reward', label=dict['method'], errorbar='se')

	plt.title(f'{dict["env"]}')
	# Add title and labels
	plt.xlabel('Episodes')
	plt.ylabel('Mean Reward')

	# Show legend
	plt.legend()

	# Show plot
	plt.savefig(f'results/{dict["env"]}/plot.pdf', format='pdf')


def env_success(env_name, score):
	if env_name == "LunarLander":
		return score > 200
	return False