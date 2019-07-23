#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class Parameters:

The class for all the model constants.
@author: Chuang, Yun-Shiuan
"""

class ModelParameter:
    MAZE_WIDTH = 12 # width of the maze
    MAZE_HEIGHT = 12 # height of the maze 
    
    # MAZE_DEPTH_TRAJECTORY = number of channels of each step in a trajectory
    # 11 = 1 (obstacle) + 1 (agent initial position) + 4 (targets) + 5 (actions)
    # in our model, 5 actions: up/down/left/right/goal
    # in the paper, also 5 actions: up/down/left/right/stay
    MAZE_DEPTH_TRAJECTORY = 11
    
    # MAZE_QUERY_STATE_DEPTH = number of channels of each query state
    # 6 = 1 (obstacle) + 1 (agent initial position) + 4 (targets)    
    MAZE_DEPTH_QUERY_STATE = 6
    
    # MAX_TRAJECTORY_SIZE = 10, number of steps of each trajectory 
    # (will be padded up/truncated to it if less/more than the constant)
    MAX_TRAJECTORY_SIZE = 10
    
    # number of layers in the resnet 
    # (5, same in the paper, A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
    NUM_RESIDUAL_BLOCKS = 5
    TRAIN_EMA_DECAY = 0.95
    WITH_PREDNET = True # True for including both charnet and prednet
    
    # Initial learning rate (LR) # paper: 10−4
    INIT_LR = 0.0001  
    # number of unique classes in the training set
    NUM_CLASS = 4 
    # the length for thr character embedding
    LENGTH_E_CHAR = 8



