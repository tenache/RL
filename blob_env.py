class BlobEnv:
    def __init__(self, original_info):
        self.original_info = original_info
        self.info = original_info
        self.total_episode_step = 0
        # self.info is the price of the dollar in pesos
        
        ## TODO: add caution_factor and other stuff in the init.  
        ### We have to change this, this should go in the init, but right now I need the original code to work, so I'll just go with this for now. 
        self.self.CAUTION_FACTOR = 0.5 # multiplies the punishment for holding
        self.self.GAMBLER_PUNISHER = 1.6 # scales the punishment for buying or selling and loosing
    def reset(self):
        self.info = self.original_info[self.total_episode_step:]
        self.episode_step = 0
        self.negative_step = 0

        return self.info
    
    # action will be one of three values. buy, sell, hold. 
    # I don't think we have the need for observation, which in the example is the state of the game. 
    # In this case, the state is simply given by the information we already have ... 
    # So, I'm not sure what to do with observation, really ... 
    # So, state is what we actually feed the model, so state is nothing other than the array we have, except that it's moved
    # one bit over every time ...
    # This will be a problem, because the size of the array will change, decisions ... decision ... 
     
    def step(self, action):
        self.episode_step += 1
        self.total_episode_step += 1
        diff = self.info[self.TIME_STEP][0] - self.info[self.TIME_STEP - 1][0]

        if action == 0: # hold 
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
            if diff >= 0:
                reward = diff/self.info[self.TIME_STEP][0]
            else:
                reward = -(diff ** self.GAMBLER_PUNISHER)/self.info[self.TIME_STEP][0]
                self.negative_step += 1
                
        else: # sell dollars
            if diff >= 0:
                reward = -(diff ** self.GAMBLER_PUNISHER)/self.info[self.TIME_STEP][0]
                self.negative_step += 1
            else:
                reward = diff/self.info[self.TIME_STEP][0]
        self.info = self.original_info[self.total_episode_step:]
        done = False
        
        # If you've accumulated 200 days with losses, time to stop ... 
        if self.episode_step  >= 300 or self.negative_step >= 150:
            done = True
        
        return self.info, reward, done