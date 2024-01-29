import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

from modified_tensorboard import ModifiedTensorBoard
from agent import DQNAgent
from blob_env import BlobEnv

from collections import deque
import random
import time
import pandas as pd
import numpy as np 
from tqdm import tqdm
import os

import argparse

REPLAY_MEMORY_SIZE = 1500
MIN_REPLAY_MEMORY_SIZE = 100
MODEL_NAME = 'FIRST_MODEL'
MINIBATCH_SIZE = 32
DISCOUNT = 1 - (1/2**6)
UPDATE_TARGET_EVERY = 5
TIME_STEP = 200
FEATURES = 3
INPUT_SHAPE = TIME_STEP, FEATURES
# self.info is the price of the dollar in pesos
CAUTION_FACTOR = 0.5 # multiplies the punishment for holding
GAMBLER_PUNISHER = 1.6 # scales the punishment for buying or selling and loosing

EPISODES = 4000
# Exploration settings 
EPSILON = 0.6 # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
ACTION_SPACE_SIZE = 3 # number of possible decisions
AGGREGATE_STATS_EVERY = 50
ep_rewards = [0]
MIN_REWARD = -1

 
def run_RL(input_shape,
           layers, 
           dropout, 
           info,
           epsilon_decay, 
           epsilon,
           min_epsilon,
           aggregate_stats_every,
           model_name):

    info_pd = pd.read_csv(info)
    info_pd = info_pd.iloc[:,1:].apply(lambda x:x.str.replace(',','.').astype(float),axis=1)
    info_arr = np.array(info_pd.iloc[:,1:])
    # For more reproducible results
    random.seed(23)
    np.random.seed(23)
    tensorflow.random.set_seed(23)
    
    if not os.path.isdir('models'):
        os.makedirs('models')
        
    # Initialize agent
    agent = DQNAgent(input_shape, layers, dropout)
    
    # Initialize environment
    env = BlobEnv(info_arr, time_step = input_shape[0])
    
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        
        # Update tensorboard step every episode
        agent.tensorboard.step = episode
    
        # Restarting episode = reset episode reward and step number
        episode_reward = 0
        step = 1
   
        # Reset environment and get initial state
        current_state = env.reset()
        
        # Reset flag and start iterating until episode ends
        done = False 
        while not done:
            
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action  ????? 
                action = np.random.randint(0, ACTION_SPACE_SIZE)
                
            new_state, reward, done = env.step(action)
            
            # Transform new continuous state to a new discrete state and count reward 
            episode_reward += reward
            
            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
            
            current_state = new_state 
            step += 1
            
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)  
            
            if not episode % aggregate_stats_every or episode == 1:
                average_reward = sum(ep_rewards[-aggregate_stats_every:])/len(ep_rewards[-aggregate_stats_every:])
                min_reward = min(ep_rewards[-aggregate_stats_every:])
                max_reward = max(ep_rewards[-aggregate_stats_every:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
                
                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= min_reward:
                    agent.model.save(f'models/{model_name}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            
            # Decay epsilon
            if  epsilon > min_epsilon:
                epsilon *= epsilon_decay
                epsilon = max(min_epsilon, epsilon)  
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_shape', type=tuple, default=(300,3))
    parser.add_argument('-l', '--layers', type=list, default=[32,32,23])
    parser.add_argument('-d', '--dropout', type=float)
    parser.add_argument('--info', type=str, default="dolar_todos.csv")
    parser.add_argument('--decay', type=float, default=0.99975)
    parser.add_argument('-e','--epsilon',type=int, default=0.6)
    parser.add_argument('--min_epsilon', type=int, default=-1)
    parser.add_argument('--aggregate_stats_every', type=int, default=50)
    parser.add_argument('--min_reward', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default='FIRST_MODEL')
    

    
    args = parser.parse_args()

    # input_shape, layers, dropout, info, epsilon_decay, epsilon, min_epsilon
    run_RL(args.input_shape,
           args.layers, 
           args.dropout,
           args.info,
           args.decay,
           args.epsilon, 
           args.min_epsilon,
           args.aggregate_stats_every,
           args.min_reward,
           args.model_name)

