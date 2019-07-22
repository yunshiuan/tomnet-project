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
import numpy as np
#import numpy as np
#from tensorflow.contrib import rnn
import commented_main_model as mm
import commented_model_parameters as mp
import commented_data_handler as dh
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
  BATCH_SIZE_PREDICT = 5
  SUBSET_SIZE = 100
  DIR_PREDICTION_ROOT = os.getcwd()
  DIR_PREDICTION_DATA_TRAJECTORY = os.path.join(DIR_PREDICTION_ROOT,'..','..',\
                                                 'data','S002a_1000files')
  DIR_PREDICTION_DATA_QUERY_STATE = DIR_PREDICTION_DATA_TRAJECTORY
#  DIR_PREDICTION_DATA_QUERY_STATE = os.path.join(DIR_PREDICTION_ROOT,'..','..',\
#                                                  'data','data_for_making_preference_predictions',\
#                                                  'query_state')
  DIR_PREDICTION_RESULT = os.path.join(DIR_PREDICTION_ROOT,'test_on_human_data')

  def __init__(self):
    pass  
  
  def parse_prediction_data_trajectory(self, directory, subset_size = -1):
    '''
    This function wil parse all the prediction files in the directoy and return 
    the corresponding trajectories.
    
    Args:
      :param directory: 
        the directory of the files to be parse.
      :param subset_size: The size of the subset (number of files) to be parsed.  
        Default to the special number -1, which means using all the files in  
        the directory. When testing the code, this could help reducing the parsing time.   
    
    Returns: 
      :prediction_data_trajectory:  
            return the 5D tensor of the whole trajectory
            (num_files, trajectory_size, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_TRAJECTORY).
      :files_prediction_trajectory:
        the names of the trajectory files being parsed.
    '''
    prediction_data_trajectory, files_prediction_trajectory = \
    self.parse_prediction_data(directory,\
                               parse_query_state = False,\
                               subset_size = subset_size)
    
    return prediction_data_trajectory, files_prediction_trajectory
  
  def parse_prediction_data_query_state(self, directory, subset_size = -1):
    '''
    This function wil parse all the prediction files in the directoy and return 
    the corresponding query states.
    
    Args:
      :param directory: 
        the directory of the files to be parse.
      :param subset_size: The size of the subset (number of files) to be parsed.  
        Default to the special number -1, which means using all the files in  
        the directory. When testing the code, this could help reducing the parsing time.   
    
    Returns: 
      :prediction_data_query_state:  
            return the 4D tensor of the whole trajectory
            (num_files, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_QUERY_STATE).
      :files_prediction_query_state:
        the names of the trajectory files being parsed.
    '''
    prediction_data_query_state, files_prediction_query_state = \
    self.parse_prediction_data(directory,\
                               parse_query_state = True,\
                               subset_size = subset_size)
    
    return prediction_data_query_state, files_prediction_query_state
  
  def parse_prediction_data(self, directory, parse_query_state, subset_size):
    '''
    This function wil parse all the prediction files in the directoy and return 
    the corresponding tensors (no need to return labels).
    
    Args:
      :param directory: 
        the directory of the files to be parse
      :param parse_query_state: 
        if 'True', parse only the query states
        and skip the actions; if 'False', parse the whole sequence 
        of trajectories    
     :param subset_size: The size of the subset (number of files) to be parsed.  
       The special number -1 means using all the files in  
       the directory. When testing the code, this could help reducing the parsing time. 
  
    Returns: 
      :prediction_data:
           if `parse_query_state == True`, 
            return the 4D tensor of the query state
            (num_files, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_QUERY_STATE);
            if `parse_query_state == False`,
            return the 5D tensor of the whole trajectory
            (num_files, trajectory_size, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_TRAJECTORY).
      :files_prediction:
        the names of the files being parsed.
    '''
    # --------------------------------------
    # List all txt files to be parsed
    # --------------------------------------
    data_handler = dh.DataHandler()
    files_prediction = data_handler.list_txt_files(directory) 
    if subset_size != -1: 
          files_prediction = files_prediction[0:subset_size]  
    # --------------------------------------
    # Print out parsing message
    # --------------------------------------
    if not parse_query_state:
      parse_mode = 'trajectories---------------'
    else:
      parse_mode = 'query states---------------'
    print('Parse ', parse_mode)
    print('Found', len(files_prediction), 'files in', directory)
    
    # --------------------------------------
    # Parse the txt files
    # --------------------------------------     
    prediction_data, _ = data_handler.parse_subset(directory,\
                                                   files_prediction,\
                                                   parse_query_state = parse_query_state)
    return prediction_data, files_prediction
  
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
      :data_set_predicted_labels:
        an array of predictions for the input data (num_files, 1).
    '''
    
    # Number of files for making predictions
    num_files_prediction = len(files_prediction)
    num_batches = num_files_prediction // batch_size
    print('%i' %num_batches, 'batches in total for making predictions ...')

    # --------------------------------------------------------------
    # Set up placeholders
    # --------------------------------------------------------------
    # Create the  placeholders          
    data_traj_placeholder = tf.placeholder(dtype=tf.float32,\
                                      shape=[batch_size,\
                                             self.MAX_TRAJECTORY_SIZE,\
                                             self.MAZE_HEIGHT,\
                                             self.MAZE_WIDTH,\
                                             self.MAZE_DEPTH_TRAJECTORY])
    # Note that this placeholder will be used only when with_prednet ==True
    data_query_state_placeholder = tf.placeholder(dtype=tf.float32,\
                                                  shape=[batch_size,\
                                                         self.MAZE_HEIGHT,\
                                                         self.MAZE_WIDTH,\
                                                         self.MAZE_DEPTH_QUERY_STATE])    
    # --------------------------------------------------------------
    # Build the graph
    # --------------------------------------------------------------
           
    if not with_prednet:
      # --------------------------------------------------------------      
      # Model with only charnet
      # -------------------------------------------------------------- 
      data_traj_placeholder = tf.placeholder(dtype=tf.float32,\
                                        shape=[batch_size,\
                                               self.MAX_TRAJECTORY_SIZE,\
                                               self.MAZE_HEIGHT,\
                                               self.MAZE_WIDTH,\
                                               self.MAZE_DEPTH_TRAJECTORY])    
      charnet = cn.CharNet()
      logits = charnet.build_charnet(data_traj_placeholder,\
                                     n=self.NUM_RESIDUAL_BLOCKS,\
                                     num_classes=self.NUM_CLASS,\
                                     reuse=True,\
                                     train=False)
      # logits = (batch_size, num_classes)
      predictions = tf.nn.softmax(logits)
      # predictions = (batch_size, num_classes)
    else:
      # --------------------------------------------------------------
      # Model with both prednet and charnet
      # --------------------------------------------------------------
      pdb.set_trace()
      charnet = cn.CharNet()
      prednet = pn.PredNet()
      length_e_char = mp.ModelParameter.LENGTH_E_CHAR
      # TODO: *** ValueError: No variables to save
      # http://jermmy.xyz/2017/04/23/2017-4-23-learn-tensorflow-save-restore-model/
      sess = tf.Session()
      saver = tf.train.Saver() 
      saver.restore(sess, self.ckpt_path)


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

    # --------------------------------------------------------------      
    # Make softmax predictions batch by batch
    # output: (num_files, num_classes)
    # --------------------------------------------------------------

    # collecting prediction_array for each batch
    # will be size of (batch_size * num_batches, num_classes)
    data_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)
    
    # Initialize unused constants
    labels_traj = np.full((batch_size,), np.nan)
    labels_query_state = np.full((batch_size,), np.nan)

    
    #pdb.set_trace()
    for step in range(num_batches):
      if step % 10 == 0:
          print('%i batches finished!' %step)
      # pdb.set_trace() 
      file_index = step * batch_size
      
      batch_data_traj, _,\
      batch_data_query_state, _\
      = mm.generate_vali_batch(data_traj,\
                               labels_traj,\
                               data_query_state,\
                               labels_query_state,\
                               batch_size,\
                               file_index = file_index)
      # pdb.set_trace()
      batch_prediction_array = sess.run(predictions,\
                                        feed_dict={data_traj_placeholder: batch_data_traj})
      # batch_prediction_array = (batch_size, num_classes)
      data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
      # data_set_prediction_array will be size of (batch_size * num_batches, num_classes),
      # or (num_files, num_classes), because num_files = sbatch_size * num_batches
      
    # --------------------------------------------------------------      
    # Make predictions based on the softmax output:
    # (num_files, num_classes) -> (num_files, 1)
    # --------------------------------------------------------------      
    data_set_predicted_labels = np.argmax(data_set_prediction_array,1)
    return data_set_predicted_labels
    
      
if __name__ == "__main__":
    # reseting the graph is necessary for running the script via spyder or other
    # ipython intepreter
    tf.reset_default_graph()
    
    preference_predictor = PreferencePredictor()
    #pdb.set_trace()
    
    # --------------------------------------------------------------      
    # parse in data for making predictions
    # --------------------------------------------------------------  
    # parse trajectory data
    prediction_data_trajectory, files_prediction_trajectory = \
    preference_predictor.parse_prediction_data_trajectory(directory = preference_predictor.DIR_PREDICTION_DATA_TRAJECTORY,\
                                                          subset_size = preference_predictor.SUBSET_SIZE)

    if preference_predictor.WITH_PREDNET:      
      # --------------------------------------------------------------      
      # model with both charnet and prednet
      # --------------------------------------------------------------  
      # parse query state data      
      prediction_data_query_state, files_prediction_query_state = \
      preference_predictor.parse_prediction_data_query_state(directory = preference_predictor.DIR_PREDICTION_DATA_QUERY_STATE,\
                                                             subset_size = preference_predictor.SUBSET_SIZE)
    else:
      # --------------------------------------------------------------      
      # model with only charnet
      # -------------------------------------------------------------- 
      prediction_data_query_state = np.full((len(files_prediction_trajectory),),\
                                             np.nan)  
    # --------------------------------------------------------------      
    # make predictions
    # output = (num_files, 1)
    # --------------------------------------------------------------      
    # pdb.set_trace()
    data_set_predicted_labels = \
    preference_predictor.predict_whole_data_set_final_targets(files_prediction_trajectory,\
                                                              prediction_data_trajectory,\
                                                              prediction_data_query_state,\
                                                              batch_size = preference_predictor.BATCH_SIZE_PREDICT,\
                                                              with_prednet = preference_predictor.WITH_PREDNET)
   




