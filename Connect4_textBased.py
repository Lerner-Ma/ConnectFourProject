# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:54:03 2023

@author: Lerner.75
"""
import numpy as np

cCount = 7
rCount = 6

blu = (0,0,255)
bk = (0,0,0)
red = (245,0,0)
ylw = (255,255,0)

def wipeBoard():
    board = np.zeros([rCount,cCount])
    return board

def getInput(board,player):
    goodInput = False
    while goodInput == False:
        #note: error with inputting other datatypes - add in check later :)
        uChoice = int(input('Which column would you like to drop your piece into? (1-7): ')) - 1
        if uChoice not in range(0,7):
            print('Invalid Input')
            
        elif board[0,uChoice] != 0: #condition where column is full
            print('Column Full! Select a Different Column')
            
        elif board[-1,uChoice] ==0: # condition where column is empty
            board[-1, uChoice] = player
            board
            #print('Empty Good Update')
            return(board)
            goodInput = True
            
        else: #column has at least one entry, not full
            for i in range(board.shape[0]-1):
                #print(i)
                if board[i+1,uChoice] != 0:
                    board[i,uChoice] = player
                    board
                    #print('Good Update')
                    return(board)
                    goodInput = True
                    
class checkStatus:
     #this class contains functions to check for vertical, horizontal, and diaganol line win conditions
     #Also checks for full board condition
     
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
    winner = max(checkStatus.Vert(board), checkStatus.Horz(board), checkStatus.diagPos(board), checkStatus.diagNeg(board))
    return(winner)               
              

gameOver = False
board = wipeBoard()
player = 1
print(board)
while gameOver == False:

    print("player:",player)
    board = getInput(board,player)
    print(board)
    
    winner = winChecker(board) #checking for win condition
    if winner != 0:
        print('Game Over! Winner:', winner)
        gameOver == True 
        break
        
    player = player%2 + 1 #alternating player 1 and 2
    
    
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/ 
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 