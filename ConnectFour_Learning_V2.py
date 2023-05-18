# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:23:46 2023

@author: Matthew
"""
'''
pip install tensorflow
pip install keras
pip install keras-rl2
'''

#https://keras.io/examples/rl/ppo_cartpole/
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import pygame as pg

cCount = 7
rCount = 6

blu = (0,0,255)
bk = (0,0,0)
red = (245,0,0)
ylw = (255,255,0)

#player = 1 #model gets first pick
#comp = 2    

class gameLogic:
     #This class contains the connect 4 game logic for environment readability
     #this class also contains functions to check for vertical, horizontal, and diaganol line win conditions
     #Also checks for full board condition
 
     #Condition checking functions:
     def Vert(board):
         winner = 0
         for j in range(board.shape[1]): # NEED TO ACCOUNT FOR CHANGES IN "SIZE" OF ARRAY RELATIVE TO POSITION
             for k in range(board.shape[0]-3):
                 nom = board[k,j]
                 #print(nom)
                 if nom > 0 and board[k+1,j] == nom and board[k+2,j] == nom and board[k+3,j] == nom:
                     winner = nom
                     #gameOver = True
                     return(winner)
         return(winner)
            
     def Horz(board):
         winner = 0
         for j in range(board.shape[1]-3):
             for k in range(board.shape[0]):
                 nom = board[k,j]
                 #print(nom)
                 if nom > 0 and board[k,j+1] == nom and board[k,j+2] == nom and board[k,j+3] == nom:
                     winner = nom
                     #gameOver = True
                     return(winner)
         return(winner)
     
     def diagPos(board):
         winner = 0
         for j in range(board.shape[1]-3):
             for k in range(board.shape[0]-3):
                 nom = board[k,j]
                 if nom > 0 and board[k+1,j+1] == nom and board[k+2,j+2] == nom and board[k+3,j+3] == nom:
                     winner = nom
                     return(winner)
         return(winner)
     
     def diagNeg(board):
        winner = 0
        for j in range(board.shape[1]-3):
            for k in range(3,board.shape[0]):
                nom = board[k,j]
                if nom > 0 and board[k-1,j+1] == nom and board[k-2,j+2] == nom and board[k-3,j+3] == nom:
                    winner = nom
                    return(winner)
        return(winner)
    
     def winChecker(board):
        winner = max(gameLogic.Vert(board), gameLogic.Horz(board), gameLogic.diagPos(board), gameLogic.diagNeg(board))
        return(winner) 
    
     def addPiece(board,uChoice, player):
         if board[0,uChoice] != 0: #condition where column is full
         #game moves on for invalid column choirce
             return(board)
         
         elif board[-1,uChoice] ==0: # condition where column is empty
             board[-1, uChoice] = player
             return(board)
         
         else: #column has at least one entry, not full
             for i in range(board.shape[0]-1):
                 #print(i)
                 if board[i+1,uChoice] != 0:
                     board[i,uChoice] = player
                     board
                     return(board)
          


class CfourEnv(Env):
    blu = (52,128,235)
    bk = (0,0,0)
    red = (242,43,12)
    ylw = (255,255,0)
        
    def __init__(self):
        #Actions include column choice for piece placement
        self.action_space = Discrete(6)
        #Observation space include game array board in binary
        self.observation_space = Box(low = 1, high = 2, shape =(rCount,cCount))
        #initial state = blank board
        self.state = np.zeros([rCount,cCount])
        #turn assignments
        self.player = random.randint(1,2)
        self.comp = self.player%2 +1
        
    def init_render(self):
        pg.init()
        squaresize = 100
        width = cCount * squaresize
        height = (rCount) * squaresize
        size = (width, height)
        self.window = pg.display.set_mode(size)
    
    def step(self, action):
        #player = 1 #model gets first pick
        #comp = 2
        reward = 0
        info = {}
        if self.player == 1:
        #adding piece to board
            self.state = gameLogic.addPiece(self.state,action,self.player)
            #check win condition, adding reward if win
            winner = gameLogic.winChecker(self.state)
            if winner == self.player :
                reward = 1
                done = True
                return self.state, reward, done, info
            elif winner != 0:
                reward = -1
                done = True
                return self.state, reward, done, info
            else:
                done = False
                
            #adding randomly generated piece for opposing player
            compAction = random.randint(0,6)
            self.state = gameLogic.addPiece(self.state,compAction,self.comp)
            winner = gameLogic.winChecker(self.state)
            if winner == self.player :
                reward = 1
                done = True
                return self.state, reward, done, info
            elif winner == self.comp:
                reward = -1
                done = True
                return self.state, reward, done, info
            else:
                done = False
                
        elif self.player == 2:
            #adding randomly generated piece for opposing player
            compAction = random.randint(0,6)
            self.state = gameLogic.addPiece(self.state,compAction,self.comp)
            winner = gameLogic.winChecker(self.state)
            if winner == self.player :
                reward = 1
                done = True
                return self.state, reward, done, info
            elif winner == self.comp:
                reward = -1
                done = True
                return self.state, reward, done, info
            else:
                done = False
                
            self.state = gameLogic.addPiece(self.state,action,self.player)
            info = {}   
            #check win condition, adding reward if win
            winner = gameLogic.winChecker(self.state)
            if winner == self.player :
                reward = 1
                done = True
                return self.state, reward, done, info
            elif winner != 0:
                reward = -1
                done = True
                return self.state, reward, done, info
            else:
                done = False
                
        
        #return step information 
        return self.state, reward, done, info

    def render(self):
        squaresize = 100
        RADIUS = int(squaresize/2 - 5)
        for c in range(cCount):
            for r in range(rCount):
               pg.draw.rect(self.window,blu,(c*squaresize,r*squaresize,squaresize,squaresize)) #drawing border rectangles
               
               #Checking if current position is occupied and by which player, and coloring accordingly
               if self.state[r][c] == 1:
                   pg.draw.circle(self.window,red,(c*squaresize+squaresize / 2, r*squaresize+squaresize / 2),RADIUS)
               elif self.state[r][c] == 2:
                   pg.draw.circle(self.window,ylw,(c*squaresize+squaresize / 2, r*squaresize+squaresize / 2),RADIUS)
               else:
                   pg.draw.circle(self.window,bk,(c*squaresize+squaresize / 2, r*squaresize+squaresize / 2),RADIUS)        
        pg.display.update()
        pg.time.delay(200)

    def reset(self):
        self.state = np.zeros([rCount,cCount])
        #turn assignments
        self.player = random.randint(1,2)
        self.comp = self.player%2 +1
        #return self.state, self.player, self.comp
        return self.state

## 
#10 random sample actions
'''
env = CfourEnv()
episodes = 10
for episode in range(1,episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode,score))
'''    
def envTest():
    ## with visualization
    env = CfourEnv()
    #env.init_render()
    episodes = 10000
    trk = []
    rngCheck = []
    for episode in range(1,episodes+1):
        state = env.reset()
        done = False
        score = 0
    
        while not done:
            #env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        #env.render()
        #pg.time.delay(2000)    
        #print('Episode:{} Score:{}'.format(episode,score))
        trk.append(score)
        rngCheck.append(env.player)
    pg.quit()
     
    print('Win rate:{}'.format((episodes-sum(trk))/episodes))

'''
###### Building Agent
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
states = env.observation_space.shape
actions = env.action_space.n

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

del model

model = build_model(states, actions)
model.summary()

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

_ = dqn.test(env, nb_episodes=15, visualize=True)

##Reloading from memory
dqn.save_weights('dqn_weights.h5f', overwrite=True)

del model
del dqn
del env

env = gym.make('CartPole-v0')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('dqn_weights.h5f')

_ = dqn.test(env, nb_episodes=5, visualize=True)
'''