'''
script for training the agent for snake using PyTorch
'''
import os
import numpy as np
from tqdm import tqdm
from collections import deque
import pandas as pd
import time
from utils import play_game, play_game2
from game_environment import Snake, SnakeNumpy
import torch
from agent_pytorch import DeepQLearningAgent
import json
# Set a spesific random seed for PyTorch to ba able to repruduce the training
torch.manual_seed(42)
version = 'v17.1'

# Loads all the training parameters from JSON
with open('model_config/{:s}.json'.format(version), 'r') as f:
    m = json.loads(f.read())
    board_size = m['board_size']
    frames = m['frames']
    max_time_limit = m['max_time_limit']
    supervised = bool(m['supervised'])
    n_actions = m['n_actions']
    obstacles = bool(m['obstacles'])
    buffer_size = m['buffer_size']

# Define how long to train and how often to log progress
episodes = 10 * (10**5)
log_frequency = 500
games_eval = 8 # Games to play when evaulating

# Create a PyTorch agent
agent = DeepQLearningAgent(board_size=board_size, frames=frames, n_actions=n_actions, 
                           buffer_size=buffer_size, version=version)

agent_type = 'DeepQLearningAgent'

# Epsilon -> the exploration rate, starts high and slowly decays
epsilon = 1
epsilon_end = 0.01
decay = 0.97   # Decay rate
reward_type = 'current' # Agent remembers teh point it got now.
sample_actions = False
n_games_training = 8*16 # Number of parralell games to run

# Start with lover epsilon if supervised training
if(supervised):
    epsilon = 0.01
    agent.load_model(file_path='models/{:s}'.format(version))

# Play some games to fill replay buffer
# The agent need some memories to learn from at the beginning
if(supervised):
    # If pretrained model existis, use it.
    try:
        agent.load_buffer(file_path='models/{:s}'.format(version), iteration=1)
    except FileNotFoundError:
        pass
else:
    # If starting from strach, play alot of games at once to have data for model.
    games = 512
    env = SnakeNumpy(board_size=board_size, frames=frames, 
                max_time_limit=max_time_limit, games=games,
                frame_mode=True, obstacles=obstacles, version=version)
    ct = time.time()
    # Play a set number of frames with high epsilon to get varied experiences to use.
    _ = play_game2(env, agent, n_actions, n_games=games, record=True,
                   epsilon=epsilon, verbose=True, reset_seed=False,
                   frame_mode=True, total_frames=games*64)


# The main training loop -------------
# Set up 2 enviorments , one for training and one for evaluation
env = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=n_games_training,
            frame_mode=True, obstacles=obstacles, version=version)
env2 = SnakeNumpy(board_size=board_size, frames=frames, 
            max_time_limit=max_time_limit, games=games_eval,
            frame_mode=True, obstacles=obstacles, version=version)

# Dict to store logs
model_logs = {'iteration':[], 'reward_mean':[],
              'length_mean':[], 'games':[], 'loss':[]}


for index in tqdm(range(episodes)):

    # Agent play a small number of states in the env to get new memories
    # The Agent uses its current epsilon to decide explore/exploit
    play_game2(env, agent, n_actions, epsilon=epsilon,
               n_games=n_games_training, record=True,
               sample_actions=sample_actions, reward_type=reward_type,
               frame_mode=True, total_frames=n_games_training, 
               stateful=True)

    # Agent performes one step of training, samples a random batch from memory and updates its weights
    loss = agent.train_agent(batch_size=64,
                             num_games=n_games_training, reward_clip=True)
    
    #Evaluate and log the progress of the agent
    # For eact "Log_Frequencies" we pause and evaluate the agents performance
    if((index+1)%log_frequency == 0):
        # Play a few games wit epsilon = -1 (turns off exploration)
        current_rewards, current_lengths, current_games = \
                    play_game2(env2, agent, n_actions, n_games=games_eval, epsilon=-1,
                               record=False, sample_actions=False, frame_mode=True, 
                               total_frames=-1, total_games=games_eval)
        # Stores the results
        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(round(int(current_rewards)/current_games, 2))
        model_logs['length_mean'].append(round(int(current_lengths)/current_games, 2))
        model_logs['games'].append(current_games)
        model_logs['loss'].append(loss)

        # Saves results to CSV file
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'length_mean', 'games', 'loss']]\
          .to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # Updates target network and Decays epsilon at set frequencies
    if((index+1)%log_frequency == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        epsilon = max(epsilon * decay, epsilon_end)