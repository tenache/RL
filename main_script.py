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

def run_RL():
    print('hello') 
    


if __name__ == '__main__':
    
    print('main')

