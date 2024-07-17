import yaml
from utils.utils import *
import importlib
import numpy as np


def read_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def instantiate_drl_method(params, wandb):
    method_name = params['name']
    module_name =  f"algos_template.{method_name.lower()}"  # Assuming module name is same as method name, just in lowercase
    #module_name =  f"algos_sol.{method_name.lower()}"
    class_name = method_name.upper()  # Assuming class name is same as method name, just in uppercase

    try:
        module = importlib.import_module(module_name)
        method_class = getattr(module, class_name)
        return method_class(params, wandb)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")


if __name__ == '__main__':
    config = read_config('config.yml')

    # Accessing global Wandb settings
    use_wandb = config.get('use_wandb', False)
    wandb_config = config.get('wandb_config', None)

    print(f"Wandb Config: {wandb_config}")
    print()

    results = []
    # Accessing parameters for each DRL method and env and performing the experiment
    for method in config['DRL_methods']:

        method['render_mode'] = method['render_mode'] if 'render_mode' in method else None

        run = {}
        run['env'] = method['gym_environment']
        run['method'] = method['name']
        run['training_episodes'] = method['tot_episodes']

        # check the params provided for the algorithm to evaluate
        check_parameters(method)

        # check if use manual or random seeds
        if len(method['seeds_to_test']) < 3:
            print("To few manual seeds provided, I\'ll generate three random ones to test the algorithm...")
            # test algorithm on random seeds
            seeds_to_test = [np.random.randint(0, 2000) for _ in range(3)]
        else:
            seeds_to_test = method['seeds_to_test']

       
        # training loop
        mean_rewards = []
        for seed in seeds_to_test:

            if use_wandb:
                wandb_config['run_name'] = method['name'] + '_seed_' + str(seed)
                wandb_config['algo'] = method['name']
                wandb_config['seed'] = seed
                wandb_config['env'] = method['gym_environment']
                wandb_config['gamma'] = method['parameters'].get('gamma', None)
                wandb_config['lr_optimizer_pi'] = method['parameters'].get('lr_optimizer_pi', None)
                wandb_config['lr_optimizer_vf'] = method['parameters'].get('lr_optimizer_vf', None)

            # instantiation of the method
            drl_method_instance = instantiate_drl_method(method, use_wandb)
            mean_rewards.append(drl_method_instance.training_loop(seed, wandb_config))

        run['mean_rewards'] = mean_rewards
        results.append(run)

    if not use_wandb:
        # plotting the results and save the figure in results/name_env/plot.png
        # Define the environment you want to filter by
        available_envs = ['CartPole-v1', 'MountainCarContinuous-v0', 'FrozenLake', 'TB3']

        for env in available_envs:
            # Filter the list based on the 'env' key
            filtered_list = [d for d in results if d.get('env') == env]
            
            if len(filtered_list) >= 1:
                print(f"\n{YELLOW_COL}\tPlotting results of env {env} in the folder: results/{env}/\n{RESET_COL}")
                plot_results(filtered_list)

    print(f"{GREEN_COL}============================================{RESET_COL}")
    print(f"{GREEN_COL} All the experiments have been performed! {RESET_COL}")
    print(f"{GREEN_COL}============================================\n\n{RESET_COL}")
