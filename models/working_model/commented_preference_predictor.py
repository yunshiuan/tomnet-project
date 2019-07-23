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
#from tensorflow.python.tools import inspect_checkpoint as chkp

#import numpy as np
#from tensorflow.contrib import rnn
import commented_main_model as mm
import commented_model_parameters as mp
import commented_data_handler as dh
import commented_charnet as cn
import commented_prednet as pn
import commented_batch_generator as bg
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
  FILE_CKPT = 'training_result/caches/cache_S030_v16_commit_???_epoch80000_tuning_batch96_train_step_1K_INIT_LR_10-4/train/model.ckpt-49'
  #FILE_CKPT = 'test_on_simulation_data/training_result/caches/cache_S030_v16_commit_926291_epoch80000_tuning_batch96_train_step_1K_INIT_LR_10-4/train/model.ckpt-999'
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
    # Restore the graph and parameters
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # --------------------------------------------------------------
    # pdb.set_trace()
    # Restore the graph from the meta graph
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    saver = tf.train.import_meta_graph(self.FILE_CKPT+'.meta')

    # Create a new session and restore the saved parameters from the checkpoint
    sess = tf.Session()
    saver.restore(sess, self.FILE_CKPT)
    print('Model restored from ', self.FILE_CKPT)

    graph = tf.get_default_graph()
    #predictions_array = (batch_size, num_classes)
    predictions_array = graph.get_tensor_by_name('train_predictions_array:0')
    
    # Inspect variables in a checkpoint
#      parameters = chkp.print_tensors_in_checkpoint_file(self.FILE_CKPT,\
#                                                         tensor_name='',\
#                                                         all_tensors=True)
    # --------------------------------------------------------------
    # Restore the placeholders
    # --------------------------------------------------------------
    train_data_traj_placeholder = graph.get_tensor_by_name('train_data_traj_placeholder:0')
    train_data_query_state_placeholder = graph.get_tensor_by_name('train_data_query_state_placeholder:0')

    # --------------------------------------------------------------      
    # Make softmax predictions batch by batch
    # output: (num_files, num_classes)
    # --------------------------------------------------------------

    # collecting prediction_array for each batch
    # will be size of (batch_size * num_batches, num_classes)
    data_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)
    
    # Initialize unused constants
    labels_traj = np.full((num_files_prediction,), np.nan)
    labels_query_state = np.full((num_files_prediction,), np.nan)

    # Create a batch generator
    batch_generator = bg.BatchGenerator()
    # pdb.set_trace()
    for step in range(num_batches):
      if step % 10 == 0:
          print('%i batches finished!' %step)
      # pdb.set_trace() 
      file_index = step * batch_size
      batch_data_traj, _,\
      batch_data_query_state, _\
      = batch_generator.generate_vali_batch(vali_data_traj = data_traj,\
                                            vali_labels_traj = labels_traj,\
                                            vali_data_query_state = data_query_state,\
                                            vali_labels_query_state = labels_query_state,\
                                            vali_batch_size = batch_size,\
                                            file_index = file_index)
      #pdb.set_trace()
      batch_prediction_array = sess.run(predictions_array,\
                                        feed_dict={train_data_traj_placeholder: batch_data_traj,\
                                                   train_data_query_state_placeholder: batch_data_query_state})
      # batch_prediction_array = (batch_size, num_classes)
      data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
      # data_set_prediction_array will be size of (batch_size * num_batches, num_classes),
      # or (num_files, num_classes), because num_files = sbatch_size * num_batches
      
    # --------------------------------------------------------------      
    # Make predictions based on the softmax output:
    # (num_files, num_classes) -> (num_files, 1)
    # --------------------------------------------------------------     
    pdb.set_trace()
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
    pdb.set_trace()
    data_set_predicted_labels = \
    preference_predictor.predict_whole_data_set_final_targets(files_prediction_trajectory,\
                                                              prediction_data_trajectory,\
                                                              prediction_data_query_state,\
                                                              batch_size = preference_predictor.BATCH_SIZE_PREDICT,\
                                                              with_prednet = preference_predictor.WITH_PREDNET)
    pdb.set_trace()
   




