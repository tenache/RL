import random as rr

class BlobEnv:
    def __init__(self, original_info, time_step, info_test):
        self.original_info_test = info_test
        self.info_test = info_test
        self.original_info = original_info
        self.info = original_info
        self.total_episode_step = 0
        self.time_step = time_step
        # self.info is the price of the dollar in pesos
        
        ## TODO: add caution_factor and other stuff in the init.  
        ### We have to change this, this should go in the init, but right now I need the original code to work, so I'll just go with this for now. 
        self.CAUTION_FACTOR = 0.5 # multiplies the punishment for holding
        self.GAMBLER_PUNISHER = 1.6 # scales the punishment for buying or selling and loosing
    
    def reset(self):
        self.start_of_window = rr.randint(0,len(self.original_info)-self.time_step*2)
        self.end_of_window = self.start_of_window + self.time_step
        self.info = self.original_info[self.start_of_window:self.end_of_window]
        self.episode_step = 0
        self.negative_step = 0
      
        return self.info
    
    def reset_test(self):
        self.start_of_window = 0
        self.end_of_window = self.start_of_window + self.time_step
        self.info = self.original_info_test[:self.end_of_window]
        self.episode_step = 0
        self.negative_step = 0
    
    # action will be one of three values. buy, sell, hold. 
    # I don't think we have the need for observation, which in the example is the state of the game. 
    # In this case, the state is simply given by the information we already have ... 
    # So, I'm not sure what to do with observation, really ... 
    # So, state is what we actually feed the model, so state is nothing other than the array we have, except that it's moved
    # one bit over every time ...
    # This will be a problem, because the size of the array will change, decisions ... decision ... 
     
    def step(self, action, test=False):

        self.episode_step += 1
        self.total_episode_step += 1
        diff = self.info[-1][0] - self.info[-2][0]
        # This is the assumption: at the start of each day you have 1 dollar and the equivalent of 1 dollar in pesos. 
        # Holding mean you keep the same amount in each currency. 
        # Because you start with 2 dollars, that's how you determine how much you've won or lost

        if action == 0: # hold 
            amount_dollars = 1
            amount_pesos = self.info[-2]/self.info[-1] 
            won  = 2 - amount_dollars + amount_pesos
            # I think this is fine, it's the immediate reward
            # I am getting
            # Holding will always give you some punishment, because unless the price is 
            # exactly the same, it means you could have benefitted from selling or buying 
            # But, we don't want to encourage the system to be wild, so we will multiply
            # this punishment by a caution factor, so the punishment will be less 
            # than if the system actually LOST the money
            reward = - abs(diff) * self.CAUTION_FACTOR
            self.negative_step += 1

        elif action == 1: # buy dollars
            won = 0
            if diff >= 0:
                reward = diff/self.info[-1][0]
                won = diff
            else:
                reward = -(abs(diff) ** self.GAMBLER_PUNISHER)/self.info[-1][0]
                self.negative_step += 1
                
        else: # sell dollars
            won = 2 - (self.info[-2]/self.info[-1]) * 2
            if diff >= 0:
                reward = -(diff ** self.GAMBLER_PUNISHER)/self.info[-1][0]
                self.negative_step += 1
            else:
                reward = diff/self.info[-1][0]
        self.start_of_window += 1
        self.end_of_window += 1
        # end_of_window = self.total_episode_step + self.time_step
        done = False
        if not test:
            self.info = self.original_info[self.start_of_window:self.end_of_window]
        else:
            self.info_test = self.original_info_test[self.start_of_window:self.end_of_window]
        
        # If you've accumulated 200 days with losses, time to stop ...
        # i think the correct is time_step 
        if self.episode_step  >= self.time_step or self.negative_step >= 150:
            done = True
        
        return self.info, reward, done, won

        