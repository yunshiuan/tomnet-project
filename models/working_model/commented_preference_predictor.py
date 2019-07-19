#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class PreferencePredictor: 

The class for making preference prediction based on a previously trained
model.
@author: Chuang, Yun-Shiuan
"""
import os
import argparse
import tensorflow as tf
#import numpy as np
#from tensorflow.contrib import rnn
import commented_main_model as mm
import commented_model_parameters as mp
import commented_charnet as cn
import commented_prednet as pn
# For debugging
import pdb
class PreferencePredictor(mp.ModelParameter): 
  
  # --------------------------------------
  # Constant: Model parameters
  # --------------------------------------
  # Use inheretance to share the model constants across classes
  
  # --------------------------------------
  # Constant: For making predictions
  # --------------------------------------
  DIR_PREDICTION_ROOT = os.getcwd()
  DIR_PREDICTION_DATA = ''
  DIR_PREDICTION_RESULT = ''

  def __init__(self):
    pass  
  def read_trained_model(self):
    pass
  
  def predict_whole_data_set_final_targets(self, files_prediction, data_traj, data_query_state, batch_size, with_prednet):
    '''
    Given one set of trajectory data and query state data,
    ask the already trained model make the predictions about the final target.
    
    Args:
      :param data_traj: 
        the trajectory data for predictions
        (num_files, max_trajectory_size, width, height, depth_trajectory)
      :param data_query_state: 
        If `with_prednet = True`, it is the query state
        of the new maze (num_files, height, width, depth_query_state).
        If `with_prednet = False`, they are ignored.
      :param batch_size: 
        the batch size
      :param with_prednet:
        If `with_prednet = True`, then construct the complete model includeing
        both charnet and prednet.
        If `with_prednet = False`, then construct the partial model including
        only the charnet.        
    Returns:
      :predictions:
        an array of predictions for the input data (num_files, ).
    '''
    
    # Number of files for making predictions
    num_files_prediction = len(files_prediction)
    num_batches = num_files_prediction // batch_size
    print('%i' %num_batches, 'batches in total for making predictions ...')

    # --------------------------------------------------------------      
    # Model with only charnet
    # --------------------------------------------------------------            
    if not with_prednet:
      # Create the image and labels_traj placeholders
      data_traj_placeholder = tf.placeholder(dtype=tf.float32,\
                                        shape=[batch_size,\
                                               self.MAX_TRAJECTORY_SIZE,\
                                               self.MAZE_HEIGHT,\
                                               self.MAZE_WIDTH,\
                                               self.MAZE_DEPTH_TRAJECTORY])    
      # --------------------------------------------------------------
      # Build the graph
      # --------------------------------------------------------------
      charnet = cn.CharNet()
      logits = charnet.build_charnet(data_traj_placeholder,\
                                     n=self.NUM_RESIDUAL_BLOCKS,\
                                     num_classes=self.NUM_CLASS,\
                                     reuse=True,\
                                     train=False)
      # logits = (batch_size, num_classes)
      predictions = tf.nn.softmax(logits)
      # predictions = (batch_size, num_classes)
  
      # --------------------------------------------------------------
      # Initialize a new session and restore a checkpoint
      # --------------------------------------------------------------
      saver = tf.train.Saver(tf.all_variables())
      sess = tf.Session()
  
      saver.restore(sess, os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
      print('Model restored from ', os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
  
      # collecting prediction_array for each batch
      # will be size of (batch_size * num_batches, num_classes)
      data_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)
      
      # collecting ground truth labels for each batch
      # will be size of (batch_size * num_batches, 1)
      data_set_ground_truth_labels = np.array([]).reshape(-1, )
      
      # Test by batches
      
      #pdb.set_trace()
      for step in range(num_batches):
        if step % 10 == 0:
            print('%i batches finished!' %step)
        # pdb.set_trace() 
        file_index = step * batch_size
        
        batch_data_traj, batch_labels_traj,\
        batch_data_query_state, batch_labels_query_state\
        = self.generate_vali_batch(data_traj,\
                                   labels_traj,\
                                   data_query_state,\
                                   labels_query_state,\
                                   batch_size,\
                                   file_index = file_index)
#        batch_data, batch_labels = self.generate_vali_batch(data, labels, batch_size, file_index)
        # pdb.set_trace()
        batch_prediction_array = sess.run(predictions,\
                                          feed_dict={data_traj_placeholder: batch_data_traj})
        # batch_prediction_array = (batch_size, num_classes)
        data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
        # vali_set_prediction_array will be size of (batch_size * num_batches, num_classes)
        data_set_ground_truth_labels = np.concatenate((data_set_ground_truth_labels, batch_labels_traj))
        
    # --------------------------------------------------------------      
    # Model with both charnet and prednet
    # --------------------------------------------------------------       
    else:
      #pdb.set_trace()
      # Create the image and labels_traj placeholders          
      data_traj_placeholder = tf.placeholder(dtype=tf.float32,\
                                        shape=[batch_size,\
                                               self.MAX_TRAJECTORY_SIZE,\
                                               self.MAZE_HEIGHT,\
                                               self.MAZE_WIDTH,\
                                               self.MAZE_DEPTH_TRAJECTORY])
      data_query_state_placeholder = tf.placeholder(dtype=tf.float32,\
                                                    shape=[batch_size,\
                                                           self.MAZE_HEIGHT,\
                                                           self.MAZE_WIDTH,\
                                                           self.MAZE_DEPTH_QUERY_STATE])    
      # --------------------------------------------------------------
      # Build the graph
      # --------------------------------------------------------------
      charnet = cn.CharNet()
      prednet = pn.PredNet()
      length_e_char = mp.ModelParameter.LENGTH_E_CHAR
      e_char = charnet.build_charnet(input_tensor = data_traj_placeholder,\
                                     n = self.NUM_RESIDUAL_BLOCKS,\
                                     num_classes = length_e_char,\
                                     reuse=True,\
                                     train=False)  
      logits = prednet.build_prednet(e_char,\
                                     data_query_state_placeholder,\
                                     n=self.NUM_RESIDUAL_BLOCKS,\
                                     num_classes = self.NUM_CLASS,\
                                     reuse=True)
      # logits = (batch_size, num_classes)
      predictions = tf.nn.softmax(logits)
      # predictions = (batch_size, num_classes)

      # --------------------------------------------------------------  
      # Initialize a new session and restore a checkpoint
      # --------------------------------------------------------------

      saver = tf.train.Saver(tf.all_variables())
      sess = tf.Session()
  
      saver.restore(sess, os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
      print('Model restored from ', os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
  
      # collecting prediction_array for each batch
      # will be size of (batch_size * num_batches, num_classes)
      data_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)
      
      # collecting ground truth labels for each batch
      # will be size of (batch_size * num_batches, 1)
      data_set_ground_truth_labels = np.array([]).reshape(-1, )
      
      # Test by batches
      #pdb.set_trace()
      for step in range(num_batches):
        if step % 10 == 0:
            print('%i batches finished!' %step)
        # pdb.set_trace() 
        file_index = step * batch_size
        
        batch_data_traj, batch_labels_traj,\
        batch_data_query_state, batch_labels_query_state\
        = self.generate_vali_batch(data_traj,\
                                   labels_traj,\
                                   data_query_state,\
                                   labels_query_state,\
                                   batch_size,\
                                   file_index = file_index)
#        batch_data, batch_labels = self.generate_vali_batch(data, labels, batch_size, file_index)
        # pdb.set_trace()
        batch_prediction_array = sess.run(predictions,\
                                          feed_dict={data_traj_placeholder: batch_data_traj,\
                                                     data_query_state_placeholder: batch_data_query_state})
        # batch_prediction_array = (batch_size, num_classes)
        data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
        # vali_set_prediction_array will be size of (batch_size * num_batches, num_classes)
        data_set_ground_truth_labels = np.concatenate((data_set_ground_truth_labels, batch_labels_traj))
      # Model with both charnet and prednet 
      
if __name__ == "__main__":
    # reseting the graph is necessary for running the script via spyder or other
    # ipython intepreter
    tf.reset_default_graph()
    
    preference_predictor = PreferencePredictor()
    
    # parse in data for making predictions
    data_traj, data_query_state, files_prediction = \
    preference_predictor.parse_prediction_data(dir_prediction = DIR_PREDICTION)

    # make predictions
    preference_predictor.predict_whole_data_set_final_targets(files_prediction,\
                                                              data_traj,\
                                                              data_query_state,\
                                                              batch_size,\
                                                              with_prednet = preference_predictor.WITH_PREDNET)
   




