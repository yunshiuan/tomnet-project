#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class BatchGenerator:

The class for making batches.
@author: Chuang, Yun-Shiuan
"""
import numpy as np

class BatchGenerator:
  def __init__(self):
    pass
  def generate_train_batch(self, train_data_traj, train_labels_traj, train_data_query_state, train_labels_query_state, train_batch_size, file_index = -1):
    '''
    This function helps you generate a batch of training data.
    
    Args:
      :param train_data_traj: 
        5D numpy array (num_files, trajectory_size, height, width, depth)
      :param train_labels_traj: 
        1D numpy array (num_files, ）
      :param train_data_query_state: 
        4D numpy array (num_files, height, width, depth)
      :param train_labels_query_state: 
        1D numpy array (num_files, ）      
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1 (default for training),
        a random batch will be chosen from the data set.
  
    Returns: 
      :train_batch_data_traj:
        a batch data. 5D numpy array (batch_size, trajectory_size, height, width, depth)
      :train_batch_labels_traj:
        a batch of labels. 1D numpy array (batch_size, )
      :train_batch_data_query_state:
        a batch data. 4D numpy array (batch_size, height, width, depth)
      :train_batch_labels_query_state:
        a batch of labels. 1D numpy array (batch_size, )        
    '''
    #pdb.set_trace()
    train_batch_data_traj, train_batch_labels_traj, train_batch_data_query_state, train_batch_labels_query_state\
    = self.generate_complete_batch(train_data_traj,\
                                   train_labels_traj,\
                                   train_data_query_state,\
                                   train_labels_query_state,\
                                   train_batch_size,\
                                   file_index)  
    return train_batch_data_traj, train_batch_labels_traj, train_batch_data_query_state, train_batch_labels_query_state
   
  def generate_vali_batch(self, vali_data_traj, vali_labels_traj, vali_data_query_state, vali_labels_query_state, vali_batch_size, file_index = -1):
    '''
    This function helps you generate a batch of validation data.
    
    Args:
      :param vali_data_traj: 
        5D numpy array (num_files, trajectory_size, height, width, depth)
      :param vali_labels_traj: 
        1D numpy array (num_files, ）
      :param vali_data_query_state: 
        4D numpy array (num_files, height, width, depth)
      :param vali_labels_query_state: 
        1D numpy array (num_files, ）      
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1 (default for validation),
        a random batch will be chosen from the data set.
  
    Returns: 
      :vali_batch_data_traj:
        a batch data. 5D numpy array (batch_size, trajectory_size, height, width, depth)
      :vali_batch_labels_traj:
        a batch of labels. 1D numpy array (batch_size, )
      :vali_batch_data_query_state:
        a batch data. 4D numpy array (batch_size, height, width, depth)
      :vali_batch_labels_query_state:
        a batch of labels. 1D numpy array (batch_size, )        
    '''
    vali_batch_data_traj, vali_batch_labels_traj, vali_batch_data_query_state, vali_batch_labels_query_state\
    = self.generate_complete_batch(vali_data_traj,\
                                   vali_labels_traj,\
                                   vali_data_query_state,\
                                   vali_labels_query_state,\
                                   vali_batch_size,\
                                   file_index)  
    return vali_batch_data_traj, vali_batch_labels_traj, vali_batch_data_query_state, vali_batch_labels_query_state

  def generate_test_batch(self, test_data_traj, test_labels_traj, test_data_query_state, test_labels_query_state, test_batch_size, file_index):
    '''
    This function helps you generate a batch of testing data.
    
    Args:
      :param test_data_traj: 
        5D numpy array (num_files, trajectory_size, height, width, depth)
      :param test_labels_traj: 
        1D numpy array (num_files, ）
      :param test_data_query_state: 
        4D numpy array (num_files, height, width, depth)
      :param test_labels_query_state: 
        1D numpy array (num_files, ）      
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1 (no default for testing),
        a random batch will be chosen from the data set.
  
    Returns: 
      :test_batch_data_traj:
        a batch data. 5D numpy array (batch_size, trajectory_size, height, width, depth)
      :test_batch_labels_traj:
        a batch of labels. 1D numpy array (batch_size, )
      :test_batch_data_query_state:
        a batch data. 4D numpy array (batch_size, height, width, depth)
      :test_batch_labels_query_state:
        a batch of labels. 1D numpy array (batch_size, )        
    '''
    test_batch_data_traj, test_batch_labels_traj, test_batch_data_query_state, test_batch_labels_query_state\
    = self.generate_complete_batch(test_data_traj,\
                                   test_labels_traj,\
                                   test_data_query_state,\
                                   test_labels_query_state,\
                                   test_batch_size,\
                                   file_index)  
    return test_batch_data_traj, test_batch_labels_traj, test_batch_data_query_state, test_batch_labels_query_state

  def generate_complete_batch(self, data_traj, labels_traj, data_query_state, labels_query_state, batch_size, file_index):
    '''
    This function helps you generate a batch of data that include both the
    trajectory data and query state data.
    
    Args:
      :param data_traj: 
        5D numpy array (num_files, trajectory_size, height, width, depth)
      :param labels_traj: 
        1D numpy array (num_files, ）
      :param data_query_state: 
        4D numpy array (num_files, height, width, depth)
      :param labels_query_state: 
        1D numpy array (num_files, ）      
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1, a random batch will be chosen from the data set.
  
    Returns: 
      :batch_data_traj:
        a batch data. 5D numpy array (batch_size, trajectory_size, height, width, depth)
      :batch_labels_traj:
        a batch of labels. 1D numpy array (batch_size, )
      :batch_data_query_state:
        a batch data. 4D numpy array (batch_size, height, width, depth)
      :batch_labels_query_state:
        a batch of labels. 1D numpy array (batch_size, )        
    '''  
    # geneate random batches for trajectories -> e_char
    batch_data_traj, batch_labels_traj = self.generate_traj_batch(data_traj, labels_traj, batch_size, file_index)
  
    # geneate random batches for query_state -> final_target
    batch_data_query_state, batch_labels_query_state = self.generate_query_state_batch(data_query_state, labels_query_state, batch_size, file_index)
    
    return batch_data_traj, batch_labels_traj, batch_data_query_state, batch_labels_query_state
  
  
  def generate_query_state_batch(self, data, labels, batch_size, file_index):
    '''
    This function helps you generate a batch of query state data
    (compared to trajectory data).
    
    Args:
      :param data: 4D numpy array (num_files, height, width, depth)
      :param labels: 1D numpy array (num_files, ）
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1, a random batch will be chosen from the data set.
  
    Returns: 
      :batch_data:
        a batch data. 4D numpy array (batch_size, height, width, depth) and 
      :batch_labels: a batch of labels. 1D numpy array (batch_size, )
    '''  
    num_files = data.shape[0]
    
    # --------------------------------------------------------------
    # Chose train_batch_size random files from all the files
    # For data:
    # data = (total_num_files, height, width, depth)->
    # batch_data = (batch_size, height, width, depth)
    #
    # For labels:
    # labels = (total_num_files,)->
    # batch_labels = (batch_size,)
    # --------------------------------------------------------------
    if file_index == -1:
      # choose a random set of files (could be not continuous)
      indexes_files = np.random.choice(num_files, batch_size)
    else:
      # choose a continuous series of files 
      indexes_files = range(file_index, file_index + batch_size)
      
    batch_data = data[indexes_files,...]
    batch_labels = labels[indexes_files,...]
    
    return batch_data, batch_labels
  
  def generate_traj_batch(self, data, labels, batch_size, file_index):
    '''
    This function helps you generate a batch of trajectory data (compared to 
    query state data).
    
    Args:
      :param data: 5D numpy array (num_files, trajectory_size, height, width, depth)
      :param labels: 1D numpy array (num_files, ）
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1, a random batch will be chosen from the data set.
  
    Returns: 
      :batch_data:
        a batch data. 5D numpy array (batch_size, trajectory_size, height, width, depth) and 
      :batch_labels: 
        a batch of labels. 1D numpy array (batch_size, )
    '''  
    # --------------------------------------------------------------
    # Paper codes
    # Generate a batch. 
    # Each example is a trejectory.
    # Each batch contains 16 examples (trajectories). Each trajectory contains 10 steps.
    # batch_data shape = (16, 10, 12, 12, 11)
    # batch_label shape = (16, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    num_files = data.shape[0]
  
    # --------------------------------------------------------------
    # Chose train_batch_size random files from all the files
    # --------------------------------------------------------------
    if file_index == -1:
      # choose a random set of files (could be not continuous)
      indexes_files = np.random.choice(num_files, batch_size)
    else:
      # choose a continuous series of files 
      indexes_files = range(file_index, file_index + batch_size)
      
    
    # --------------------------------------------------------------    
    # Choose the data and labels coresponding to the indexes files
    # For data:
    # data = (total_num_files, num_steps, height, width, depth)->
    # batch_data = (batch_size, num_steps, height, width, depth)->
    # For labels:
    # labels = （num_files, ） ->
    # batch_labels = (batch_size, )
    # --------------------------------------------------------------   
    batch_data = data[indexes_files,...]
    batch_labels = labels[indexes_files]
    # pdb.set_trace()
    assert(batch_labels.shape[0]==batch_size)
    #pdb.set_trace()
    return batch_data, batch_labels
