# script to visualize how agent plays a game
# useful to study different iterations

import numpy as np
import pandas as pd
from agent_pytorch import DeepQLearningAgent
from game_environment import Snake, SnakeNumpy
from utils import visualize_game
import json


def find_best_iteration(log_file):
    """
    Reads training log CSV and returns the iteration number with the highest mean reward. 
    (Best model found after training)
    """
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        return None

    if df.empty or 'reward_mean' not in df.columns:
        return None

    # Find best index of max(reward_mean)
    best_idx = df['reward_mean'].idxmax()
    
    # Get the data from that best-performing row
    best_performance_row = df.loc[best_idx]
    
    best_iteration = int(best_performance_row['iteration'])
    best_reward = best_performance_row['reward_mean']
    
    return best_iteration

# some global variables
version = 'v17.1'

with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames'] # keep frames >= 2
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])

best_iteration = find_best_iteration("model_logs/v17.1.csv")

iteration_list = [1000, best_iteration]
max_time_limit = 398

# setup the environment
env = Snake(board_size=board_size, frames=frames, max_time_limit=max_time_limit,
            obstacles=obstacles, version=version)
s = env.reset()
n_actions = env.get_num_actions()

# setup the agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, 
                           n_actions=n_actions, buffer_size=10, version=version)
# agent = PolicyGradientAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = AdvantageActorCriticAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = HamiltonianCycleAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)
# agent = BreadthFirstSearchAgent(board_size=board_size, frames=frames, n_actions=n_actions, buffer_size=10)

for iteration in iteration_list:
    agent.load_model(file_path='models/{:s}'.format(version), iteration=iteration)
    
    for i in range(5):
        visualize_game(env, agent,
            path='images/game_visual_{:s}_{:d}_14_ob_{:d}.mp4'.format(version, iteration, i),
            debug=False, animate=True, fps=12)
