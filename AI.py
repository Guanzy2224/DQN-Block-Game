import numpy as np
import sys
from termcolor import colored


# Define game environment
class game:
    def __init__(self, size = 10):
        # Define different shape of blocks
        b1 = [[0,0]] 
        b2 = [[0,0],[1,0]] 
        b3 = [[0,0],[0,1]]
        b4 = [[0,0],[0,1],[1,0]]
        b5 = [[0,0],[0,1],[1,1]]
        b6 = [[0,0],[1,1],[1,0]]
        b7 = [[1,1],[0,1],[1,0]]
        b8 = [[0,0],[0,1],[0,2]]
        b9 = [[0,0],[1,0],[2,0]]
        b10 = [[0,0],[0,1],[0,2],[0,3],[0,4]]
        b11 = [[0,0],[1,0],[2,0],[3,0],[4,0]]
        b12 = [[0,0],[0,1],[0,2],[1,2],[2,2]]
        b13 = [[0,0],[1,0],[2,0],[2,1],[2,2]]
        b14 = [[0,0],[0,1],[0,2],[1,0],[2,0]]
        b15 = [[2,0],[2,1],[2,2],[1,2],[0,2]]
        b16 = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
        b17 = [[0,0],[0,1],[0,2],[1,2]]
        b18 = [[1,0],[1,1],[1,2],[0,2]]
        b19 = [[0,0],[1,0],[2,0],[2,1]]
        b20 = [[0,1],[1,1],[2,1],[2,0]]
        b21 = [[0,1],[1,0],[1,1],[1,2],[2,1]]
        b22 = [[0,0],[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[2,2]]
        b23 = [[0,0],[0,1],[0,2],[0,3]]
        b24 = [[0,0],[1,0],[2,0],[3,0]]
        b25 = [[0,0],[1,0],[0,1],[1,1]]

        # Bigger and more complex block will increase game difficulty
        # Current version performs well under difficulty 1 and 2
        difficulty_1 = [b1,b2,b3,b8,b9]
        difficulty_2 = [b1,b2,b3,b4,b5,b6,b7,b8,b9]
        difficulty_3 = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,                        b23,b24,b25,
                        b1,b2,b3,b4,b5,b6,b7,b8,b9,                                                    b23,b24,
                        b1,b2,b3,b4,b5,b6,b7,b8,b9]
        difficulty_4 = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,    b17,b18,b19,b20,        b23,b24]
        difficulty_5 = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24]

        self.all_blocks = difficulty_2
        self.bolck_kinds = len(self.all_blocks)
        self.size = size
        self.map = np.zeros((self.size,self.size))
        self.blocks = self.all_blocks
        self.block_counts = len(self.all_blocks)
        self.block_list = [] # The current block that AI need to choose a place to put, in xy format
        self.block_list_map = [] # The current block that AI need to choose a place to put, in map format
        
    # Transfer blocks from xy format to map format
    def xy_to_shape(self, xy):
        minimap = np.zeros((5,5))
        for i, j in xy:
            minimap[i, j] = 1
        return minimap

    # Randomly choose a block as the next block
    def generate_block(self):
        self.block_list = list(np.random.choice(self.blocks, size=1))
        self.block_list_map = self.xy_to_shape(self.block_list[0])

    # Decide whether current block can be legitly placed in this position
    def is_fit(self, block, i0, j0, curr_map = []):
        ans = True
        if curr_map == []:
            curr_map = self.map

        for (i, j) in block:
            if i0+i >= self.size:
                ans = False
                break
            if j0+j >= self.size:
                ans = False
                break
            if curr_map[i0+i, j0+j] == 1:
                ans = False
                break

        return ans

    # Return a map that has the current block added to the current game map
    def add_box_to_map(self, block0, ii0, jj0, curr_map=[]):
        if curr_map == []:
            curr_map = np.copy(self.map)
        if not self.is_fit(block0, ii0, jj0):
            return([])
        else:
            newmap0 = np.copy(curr_map)
            for i, j in block0:
                newmap0[ii0+i, jj0+j] = 1

            newmap = np.copy(newmap0)
            for i in range(self.size):
                if np.sum(newmap0[i,:]) == self.size:
                    newmap[i,:] = 0
                if np.sum(newmap0[:,i]) == self.size:
                    newmap[:,i] = 0
        
            return newmap

    # Update the game map with the new map which block is just added
    def map_update(self, block0, ii0, jj0, printact = True):
        if not self.is_fit(block0, ii0, jj0):
            return(False)
        else:
            newmap = self.add_box_to_map(block0, ii0, jj0)
            if printact:
                self.print_act(old_map = self.map, new_map = newmap)
            self.map = newmap
    
    # Print the action of adding block
    def print_act(self, old_map, new_map):
        for i in range(self.size):
            for j in range(self.size):
                if new_map[i,j] == 0:
                    if new_map[i,j] == old_map[i,j]:
                        sys.stdout.write(" ." )
                    else:
                        sys.stdout.write(colored(" *", "red" ))
                else:
                    if new_map[i,j] == old_map[i,j]:
                        sys.stdout.write(unichr(0x2588)*2)
                    else:
                        sys.stdout.write(colored(unichr(0x2588)*2, "red"))
            sys.stdout.write('\n')
        print('------')

    # Print current game map
    def print_board(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i,j] == 0:
                    sys.stdout.write(" ." )
                else:
                    sys.stdout.write(unichr(0x2588)*2)
            sys.stdout.write('\n')

    # Print current block in block_list
    def print_blocks(self):
        block_map = self.block_list_map
        for i in range(5):
            for j in range(5):
                if block_map[i,j] == 0:
                    sys.stdout.write('  ')
                else:
                    sys.stdout.write(colored(unichr(0x2588)*2, "red"))
            sys.stdout.write('\n')
            

# This function is to test the game environment, nothing to do with RL model
def play_by_hand(steps=10):
    board = game(10)
    for _ in range(steps):
        board.print_board()
        board.generate_block()
        board.print_blocks()
        xi = int(raw_input('Row i:'))
        yj = int(raw_input('Col j:'))
        board.map_update(board.block_list[0], ii0=xi, jj0=yj)


# RL model begin, the following lines will be better saved in another file
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Flatten
import random

# agent = AI
class agent():

    def __init__(self, side_size=10):
        # Game parameters
        self.side_size = side_size
        self.board_size = side_size * side_size

        # Reinforcement learning parameters
        self.gamma =  1.0 # Discount rate
        self.alpha = 0.5 # Learning rate in bellman equation
        self.epochs = 10000 # Deprecated
        self.epsilon = 1.2 # Learning rate of randomly choosing an act
        self.epsilon_decay = 0.998 # Learning rate decay as time goes by
        self.epsilon_min = 0.05 # Minimum learning rate to prevent tracking in local maximum
        self.max_step = 1000 # Max step to play the game

        # Q-function (NN) setup
        self.model = Sequential()
        self.model.add(Flatten())
        self.model.add(Dense(units=(self.board_size) * 8, input_dim=self.board_size, activation="relu"))
        self.model.add(Dense(units=self.board_size*8, activation="relu"))
        self.model.add(Dense(units=self.board_size*2, activation="relu"))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))
        self.tfboard = keras.callbacks.TensorBoard(log_dir='logdir',update_freq=2)
        
        # Keep records for analysis
        self.lossrecord = []
        self.steprecord = []

    # Given a game map and a block, generate all possible new map(s),
    # In each map the block is placed in a different place 
    def draft_map(self, map, block, board):
        ans = []
        xy = []
        for i0 in range(self.side_size):
            for j0 in range(self.side_size):
                newmap = board.add_box_to_map(block, i0, j0, map)
                if len(newmap) >= 1:
                    ans.append(newmap)
                    xy.append([i0,j0])
        return(ans, xy)

    # Using Deep-Q Learning framework to train
    def train(self, episode=200):
        for _ in range(episode): # One episode = one game
            board = game(self.side_size) # set up a new game board
            finished = False # if finished, end this loop
            temp_loss = [] # record loss in each step in this game

            for step in range(self.max_step):
                board.generate_block() # push a new block to the block_list
                next_states, next_xy = self.draft_map(board.map, board.block_list[0], board) # Generate next states
                if len(next_states) == 0: # when nowhere to place the block
                    self.model.fit(x=np.array([board.map]), y=np.array([-100]), epochs=1, verbose=0)
                    finished = True
                else:
                    Q0 = self.model.predict(np.array(next_states)) # find the best(highest value) place to put
                    count0 = np.sum(board.map) # using the count of grids being taken to calculate reward
                    if random.uniform(0,1)<self.epsilon: # generate a random action
                        myact = random.sample(next_xy,1)[0]
                        board.map_update(board.block_list[0], myact[0], myact[1], printact = False) # Change to True to see each step
                        if self.epsilon > self.epsilon_min:
                            self.epsilon *= self.epsilon_decay
                    else: # choose best action psuggested by Q value
                        myact = next_xy[np.argmax(Q0)]
                        board.map_update(board.block_list[0], myact[0], myact[1], printact = False) # Change to True to see each step
                    count1 = np.sum(board.map)
                    reward = max(0, count0 - count1) # TODO: would quadratic reward be better?
                    mynext_state = next_states[np.argmax(Q0)] 
                    Q_ = 0 # Q of next step will be calculated by summing up the Q in each scenario
                    for next_block in board.all_blocks:
                        next2_states, next2_xy = self.draft_map(mynext_state, next_block, board)
                        if len(next2_states) == 0:
                            Q_ += -200 / board.bolck_kinds # When next step is dead end, the Q is set to -200
                        else:
                            Q_ += np.max(self.model.predict(np.array(next2_states)))/board.bolck_kinds # otherwise next step choosing optimum act
                    Q0_new = np.array(( 1 - self.alpha ) * np.max( Q0 ) + self.alpha * ( reward + self.gamma * Q_ ) ) # Targeted Q for training


                    hist = self.model.fit(x=np.array([mynext_state]), y=np.array([Q0_new]), epochs=1, verbose=0) #,callbacks=[self.tfboard])
                    temp_loss.extend(hist.history['loss']) # Record the loss in this step
                    

                if finished:
                    print('------------died at:' + str(step) + ' ---------------')
                    self.steprecord.append(step)
                    self.lossrecord.append(temp_loss)
                    break
            if step==1000:
                print('------------died at: 1000 ---------------')
                self.steprecord.append(1000)
                self.lossrecord.append(temp_loss)
                
                        
myAgent = agent(side_size=10)
myAgent.train()

# Save trained model
import pickle
pickle.dump(myAgent, open( "save.p", "wb" ) )