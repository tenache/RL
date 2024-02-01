from modified_tensorboard import ModifiedTensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

from collections import deque
import tensorflow
import time
import random
import numpy as np

REPLAY_MEMORY_SIZE = 1500
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 32
DISCOUNT = 1 - (1/2**6) 
UPDATE_TARGET_EVERY = 5

class DQNAgent:
    def __init__(self, input_shape_, layers, dropout, model_name):
        # Main model
        # gets trained every step
        self.model = self.create_model(input_shape_, layers, dropout)

        # Target network
        # .predict every step
        # every n steps, we update the model that we've been fitting for every step, and I guess we discard the old one ... 
        self.target_model = self.create_model(input_shape_, layers, dropout)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_name, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0 
        
    def create_model(self, input_shape_, layers, dropout):
        model = Sequential()
        model.add(LSTM(layers[0],  input_shape=input_shape_))
        
        for i in range(1,len(layers)):
            if dropout:
                model.add(Dropout(dropout))
            model.add(Dense(layers[i], activation="relu"))
        model.add(Dense(3, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model
   
    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
                # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        # for transition in minibatch:
            # print(np.shape(transition[3]))
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        
        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0