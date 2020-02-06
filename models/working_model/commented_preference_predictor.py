#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class PreferencePredictor:

The class for making preference prediction based on a previously trained
model.
Note that the input data should be carefully designed so that
all targets are of the same distance to the agent. By doing this,
the physical distance will be cancelled out and the social distance will
be the only remaining factor. Thus, the 'prediction proportion'/'avg_prediction_probability'
will equal to 'preference score'.
@author: Chuang, Yun-Shiuan
"""
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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
  # param
  BATCH_SIZE_PREDICT = 16
  SUBSET_SIZE = 96
  BREAK_CORRESPONDENCE = True # This should be True when using the same set of files for both trajectory and query state data to avoid overestimating the accuracy.
  VERSION = 'Traj_S003b_Query_Stest_subset96'
  WITH_LABEL = False # whether the query state has final target label
#  VERSION = 'Traj_S003b_Query_S003b_subset96'
  # dir
  DIR_PREDICTION_ROOT = os.getcwd() # the script dir

  # For human data----------
#  DIR_PREDICTION_DATA_TRAJECTORY = os.path.join(DIR_PREDICTION_ROOT,'..','..',\
#                                                'data','data_human','processed','S030')
#  DIR_PREDICTION_DATA_QUERY_STATE = os.path.join(DIR_PREDICTION_ROOT,'..','..',\
#                                                'data','data_preference_predictions',\
#                                                'query_state')
#
#  DIR_MODEL = 'test_on_human_data/training_result/caches/cache_S030_v9_commit_78092b'

  # For simulation data-----------
  DIR_PREDICTION_DATA_TRAJECTORY = os.path.join(DIR_PREDICTION_ROOT,'..','..',\
                                                 'data','data_simulation','S003b')
#  DIR_PREDICTION_DATA_QUERY_STATE = DIR_PREDICTION_DATA_TRAJECTORY
  DIR_PREDICTION_DATA_QUERY_STATE = os.path.join(DIR_PREDICTION_ROOT,'..','..',\
                                                'data','data_preference_predictions',\
                                                'query_state')
  DIR_MODEL = 'test_on_simulation_data/training_result/caches/cache_S003b_v24_commit_014d79'
  # --------------------

  DIR_MODEL_PREDICTION_RESULT_ROOT = os.path.join(DIR_MODEL,'prediction')
  DIR_MODEL_PREDICTION_RESULT_THIS_VERSION = \
  os.path.join( DIR_MODEL_PREDICTION_RESULT_ROOT,VERSION)

  if not os.path.exists(DIR_MODEL_PREDICTION_RESULT_THIS_VERSION):
    os.makedirs(DIR_MODEL_PREDICTION_RESULT_THIS_VERSION)

  # file
  FILE_MODEL_CKPT = os.path.join(DIR_MODEL,'train','model.ckpt-9999')
  #FILE_MODEL_CKPT = 'test_on_simulation_data/training_result/caches/cache_S030_v16_commit_926291_epoch80000_tuning_batch96_train_step_1K_INIT_LR_10-4/train/model.ckpt-999'

  def __init__(self):
    pass

  def parse_prediction_data_whole_data_set(self):
    '''
    This encapsulated function wil parse all the prediction files in the directory and
    return both the tensors, labels file names of corresponding
    trajectories and query states.
    '''
    # --------------------------------------------------------------
    # parse in data for making predictions
    # --------------------------------------------------------------
    # parse trajectory data
    self.prediction_data_trajectory, self.prediction_data_ground_truth_labels_traj, self.files_total_traj = \
    preference_predictor.parse_prediction_data_trajectory(directory = preference_predictor.DIR_PREDICTION_DATA_TRAJECTORY,\
                                                          subset_size = preference_predictor.SUBSET_SIZE)

    if self.WITH_PREDNET:
      # --------------------------------------------------------------
      # model with both charnet and prednet
      # --------------------------------------------------------------
      # parse query state data
      self.prediction_data_query_state, self.prediction_data_ground_truth_labels_query_state, self.files_total_query_state = \
      preference_predictor.parse_prediction_data_query_state(directory = preference_predictor.DIR_PREDICTION_DATA_QUERY_STATE,\
                                                             subset_size = preference_predictor.SUBSET_SIZE,\
                                                             break_correspondence = preference_predictor.BREAK_CORRESPONDENCE,\
                                                             with_label = self.WITH_LABEL)
      # ground truth labels for the final targets
      self.final_target_ground_truth_labels = self.prediction_data_ground_truth_labels_query_state

    else:
      # --------------------------------------------------------------
      # model with only charnet
      # --------------------------------------------------------------
      self.prediction_data_query_state = np.full((len(self.prediction_data_query_state),),\
                                                 np.nan)
      # ground truth labels for the final targets
      self.final_target_ground_truth_labels = self.prediction_data_ground_truth_labels_traj

  def parse_prediction_data_trajectory(self, directory, subset_size = -1):
    '''
    This function wil parse all the prediction files in the directory and return
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
      :prediction_data_ground_truth_labels_traj:
        the ground truth labels for the query state (num_files, 1).
      :files_total_traj:
        the names of the trajectory files being parsed.
    '''
    prediction_data_trajectory, prediction_data_ground_truth_labels_traj,\
    files_total_traj = \
    self.parse_prediction_data_subset(directory,\
                               parse_query_state = False,\
                               subset_size = subset_size)

    return prediction_data_trajectory, prediction_data_ground_truth_labels_traj, files_total_traj

  def parse_prediction_data_query_state(self, directory, subset_size = -1, break_correspondence = False, with_label = True):
    '''
    This function wil parse all the prediction files in the directory and return
    the corresponding query states.

    Args:
      :param directory:
        the directory of the files to be parse.
      :param subset_size: The size of the subset (number of files) to be parsed.
        Default to the special number -1, which means using all the files in
        the directory. When testing the code, this could help reducing the parsing time.
      :param break_correspondence:
        Whether or not to break the correspondence between trajectory files and
        query state files. This should be True when using the same set of files
        for both trajectory and query state data to avoid overestimating the
        accuracy. If `break_correspondence = True`, the query state data
        will be shuffled randomly. (Default as False)

    Returns:
      :prediction_data_query_state:
            return the 4D tensor of the whole trajectory
            (num_files, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_QUERY_STATE).
      :prediction_data_ground_truth_labels_query_state:
        the ground truth labels for the query state (num_files, 1).
      :files_total_query_state:
        the names of the trajectory files being parsed.
    '''
    prediction_data_query_state, prediction_data_ground_truth_labels_query_state,\
    files_total_query_state = \
    self.parse_prediction_data_subset(directory,\
                               parse_query_state = True,\
                               subset_size = subset_size,\
                               with_label = with_label)
    if break_correspondence:
      # random_state = 0 -> Make the result reproducible
      prediction_data_query_state,\
      prediction_data_ground_truth_labels_query_state,\
      files_total_query_state = \
      shuffle(prediction_data_query_state,\
              prediction_data_ground_truth_labels_query_state,\
              files_total_query_state,\
              random_state = 0)
      # pdb.set_trace()

    return prediction_data_query_state, prediction_data_ground_truth_labels_query_state, files_total_query_state

  def parse_prediction_data_subset(self, directory, parse_query_state, subset_size, with_label = True):
    '''
    This function wil parse either the trajectory-type prediction files
    or the query-state-type prediction files in the directory and
    return the corresponding tensors.

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
     :param with_label:
       whether the query state file contains the final
       target (default to True). Note that this will be False for
       preference inference on equal-distance files since there are no
       final tagrgets in such files.

    Returns:
      :prediction_data:
           if `parse_query_state == True`,
            return the 4D tensor of the query state
            (num_files, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_QUERY_STATE);
            if `parse_query_state == False`,
            return the 5D tensor of the whole trajectory
            (num_files, trajectory_size, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_TRAJECTORY).
      :ground_truth_labels:
        the ground truth labels (num_files, 1)
      :files_prediction:
        the names of the files being parsed.
    '''
    # --------------------------------------
    # List all txt files to be parsed
    # - input for data_handler.parse_subset()
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
    prediction_data, ground_truth_labels = data_handler.parse_subset(directory,\
                                                                     files_prediction,\
                                                                     parse_query_state = parse_query_state,\
                                                                     with_label = with_label)
    return prediction_data, ground_truth_labels, files_prediction

  def predict_preferences(self):
    '''
    The encapusulated function of predict_whole_data_set_final_targets()
    '''
    self.prediction_proportion, self.ground_truth_label_proportion,\
    self.prediction_count, self.averaged_predicted_probability,\
    self.ground_truth_label_count,\
    self.data_set_predicted_labels, self.data_set_ground_truth_labels,\
    self.files_prediction_traj, self.files_prediction_query_state= \
    self.predict_whole_data_set_final_targets(files_total_traj = self.files_total_traj,\
                                              files_total_query_state = self.files_total_query_state,\
                                              data_traj = self.prediction_data_trajectory,\
                                              data_query_state = self.prediction_data_query_state,\
                                              final_target_ground_truth_labels = self.final_target_ground_truth_labels,\
                                              batch_size = self.BATCH_SIZE_PREDICT,\
                                              with_prednet = self.WITH_PREDNET,
                                              with_label = self.WITH_LABEL)

  def predict_whole_data_set_final_targets(self, files_total_traj, files_total_query_state, data_traj, data_query_state, final_target_ground_truth_labels, batch_size, with_prednet, with_label = True):
    '''
    Given one set of trajectory data and query state data,
    ask the already trained model make the predictions about the final target.

    Args:
      :param files_total_traj:
        the total trajectory files for making predictions.
        Note that some files will not be used
        as they are remainders when dividing files_total by batch_size.
      :param files_total_query_state:
        the total query_state files for making predictions.
        Note that some files will not be used
        as they are remainders when dividing files_total by batch_size.
      :param data_traj:
        the trajectory data for predictions
        (num_files, max_trajectory_size, width, height, depth_trajectory)
      :param data_query_state:
        If `with_prednet = True`, it is the query state
        of the new maze (num_files, height, width, depth_query_state).
        If `with_prednet = False`, they are ignored.
      :param final_target_ground_truth_labels:
        the ground truth labels for the final targets (num_files, 1).
      :param batch_size:
        the batch size
      :param with_prednet:
        If `with_prednet = True`, then construct the complete model includeing
        both charnet and prednet.
        If `with_prednet = False`, then construct the partial model including
        only the charnet.
      :param with_label:
        whether the query state file contains the final
        target (default to True). Note that this will be False for
        preference inference on equal-distance files since there are no
        final tagrgets in such files.
    Returns:
      :prediction_proportion:
        an array of the frequncey of predictions (num_classes, 1).
      :ground_truth_label_proportion:
        an array of the frequncey of ground truth labels (num_classes, 1).
      :prediction_count:
        an array of the count of predictions (num_classes, 1).
      :averaged_predicted_probability:
        the averaged predicted probability of each target across all mazes
        (num_classes, 1)
      :ground_truth_label_count:
        an array of the count of predictions (num_classes, 1).
      :data_set_predicted_labels:
        an array of predictions for the input data (num_batch * batch_size, 1).
        Note that (num_batch * batch_size) might not equal to num_files
        because the num_files might not be divisible by batch_size.
      :data_set_ground_truth_labels:
        an array of ground truth labels for the input data (num_batch * batch_size, 1).
        Note that (num_batch * batch_size) might not equal to num_files
        because the num_files might not be divisible by batch_size.
      :files_prediction_traj:
        the trajectory files used for making predictions (num_batch * batch_size).
      :files_prediction_query_state:
        the query state files used for making predictions (num_batch * batch_size).
    '''
    # pdb.set_trace()
    # Number of files for making predictions
    assert len(files_total_traj) == len(files_total_query_state)

    num_files_total = len(files_total_traj)
    num_batches = num_files_total // batch_size # exclude the remainder files
    num_files_prediction = num_batches * batch_size
    print('%i' %num_batches, 'batches in total for making predictions ...')

    # --------------------------------------------------------------
    # Restore the graph
    # --------------------------------------------------------------
    # pdb.set_trace()
    graph, sess, train_data_traj_placeholder, train_data_query_state_placeholder,\
    predictions_array = self.restore_graph()

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
      batch_data_query_state, batch_labels_query_state\
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

    # --------------------------------------------------------------
    # Make predictions based on the softmax output:
    # (num_files, num_classes) -> (num_files, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    data_set_predicted_labels = np.argmax(data_set_prediction_array, 1)

    # --------------------------------------------------------------
    # Derive the frequncey of predictions for each target :
    # (num_files, num_classes) -> (num_classes, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    prediction_count = np.zeros(self.NUM_CLASS)
    prediction_count_detail = np.unique(data_set_predicted_labels,return_counts=True)
    # use loop to fill in the value in case the number of unique
    # labels in less than the NUM_CLASS
    for class_index in range(0,len(prediction_count_detail[0])):
      class_name = prediction_count_detail[0][class_index]
      prediction_count[class_name] = prediction_count_detail[1][class_index]

    prediction_proportion = np.round(prediction_count/np.sum(prediction_count), 2)


    # --------------------------------------------------------------
    # Derive the averaged predicted probability of each target across all mazes :
    # (num_files, num_classes) -> (num_classes, 1)
    # --------------------------------------------------------------
    averaged_predicted_probability = np.round(np.mean(data_set_prediction_array, 0),2)


    # --------------------------------------------------------------
    # Set ground truth labels as final target labels:
    # (num_files_total, 1) -> (num_files_prediction, 1)
    # --------------------------------------------------------------
    data_set_ground_truth_labels = final_target_ground_truth_labels[0:num_files_prediction]

    # --------------------------------------------------------------
    # Derive the frequncey of ground truth labels for each target :
    # (num_files, num_classes) -> (num_classes, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    if with_label:
      ground_truth_label_count = np.unique(data_set_ground_truth_labels,return_counts=True)[1]
      ground_truth_label_proportion = np.round(ground_truth_label_count/np.sum(ground_truth_label_count), 2)
    else:
      ground_truth_label_count = np.zeros((self.NUM_CLASS))
      ground_truth_label_proportion = np.zeros((self.NUM_CLASS))
    # --------------------------------------------------------------
    # Return the files used for making predictions
    # (num_files_total, 1) -> (num_files_prediction, 1)
    # --------------------------------------------------------------
    files_prediction_traj = files_total_traj[0:num_files_prediction]
    files_prediction_query_state = files_total_query_state[0:num_files_prediction]

    return  prediction_proportion, ground_truth_label_proportion, prediction_count, averaged_predicted_probability, ground_truth_label_count, data_set_predicted_labels, data_set_ground_truth_labels, files_prediction_traj, files_prediction_query_state

  def restore_graph(self):
    '''
    Restore the graph and parameters from a checkpoint.

    Returns:
      :graph:
        the restored graph.
      :sess:
        the session with the restored graph.
      :train_data_traj_placeholder:
        the placeholder for trajectory input data in the graph
        (batch_size, max_trajectory_size, width, height, depth_trajectory).
      :train_data_query_state_placeholder:
        the placeholder for the query state input data in the graph
        (batch_size, width, height, depth_query_state).
      :predictions_array:
        the placeholder for the prediction array output
        (batch_size, num_classes).
    '''

    # --------------------------------------------------------------
    # Restore the graph and parameters
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # --------------------------------------------------------------
    # pdb.set_trace()
    # Restore the graph from the meta graph
    # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    saver = tf.train.import_meta_graph(self.FILE_MODEL_CKPT+'.meta')

    # Create a new session and restore the saved parameters from the checkpoint
    sess = tf.Session()
    saver.restore(sess, self.FILE_MODEL_CKPT)
    print('Model restored from ', self.FILE_MODEL_CKPT)

    graph = tf.get_default_graph()
    #predictions_array = (batch_size, num_classes)

    # --------------------------------------------------------------
    # Restore the output placeholders
    # --------------------------------------------------------------
    predictions_array = graph.get_tensor_by_name('train_predictions_array:0')

    # --------------------------------------------------------------
    # Restore the input placeholders
    # --------------------------------------------------------------
    train_data_traj_placeholder = graph.get_tensor_by_name('train_data_traj_placeholder:0')
    train_data_query_state_placeholder = graph.get_tensor_by_name('train_data_query_state_placeholder:0')


    # Inspect variables in a checkpoint
#      parameters = chkp.print_tensors_in_checkpoint_file(self.FILE_MODEL_CKPT,\
#                                                         tensor_name='',\
#                                                         all_tensors=True)
    return graph, sess, train_data_traj_placeholder, train_data_query_state_placeholder, predictions_array

  def save_predictions(self):
    '''
    The encapusulated function to save the predictions, including
      (1) prediction_proportion (num_classes, 1)
      (2) data_set_predicted_labels (num_files, 1)
      (3) data_set_ground_truth_labels (num_files, 1)

    output:
      (1) final_target_predictions.csv
      (2) proportion_prediction_and_ground_truth_labels.csv
    '''
    # pdb.set_trace()

    # --------------------------------------------------------------
    # collect the predictions
    # --------------------------------------------------------------
    # the predictions about the final targets
    correctness = np.equal(self.data_set_ground_truth_labels,\
                           self.data_set_predicted_labels)

    df_final_target_predictions = pd.DataFrame(data = {'files_trajectory': self.files_prediction_traj,\
                                                       'files_query_state': self.files_prediction_query_state,\
                                                       'final_target_ground_truth_labels': self.data_set_ground_truth_labels,\
                                                       'final_target_predicted_labels': self.data_set_predicted_labels,\
                                                       'correctness': correctness.astype(int)})
    # the frequncey of predictions the ground truth labels for each target
    # pdb.set_trace()
    df_proportion_prediction_and_ground_truth_labels = pd.DataFrame(data = {'targets': range(0,4),\
                                                                           'ground_truth_label_proportion': self.ground_truth_label_proportion,\
                                                                           'prediction_proportion': self.prediction_proportion,\
                                                                           'avg_prediction_probability': self.averaged_predicted_probability,\
                                                                           'ground_truth_label_count': self.ground_truth_label_count,\
                                                                           'prediction_count': self.prediction_count,\
                                                                           'accuracy_data_set': np.round((sum(correctness)/len(correctness))*100,2)
                                                                           })
    # --------------------------------------------------------------
    # write csv files
    # --------------------------------------------------------------
    # the predictions about the final targets
    file_name_final_target_predictions = os.path.join(self.DIR_MODEL_PREDICTION_RESULT_THIS_VERSION,\
                                                      'final_target_predictions.csv')
    df_final_target_predictions.to_csv(file_name_final_target_predictions)

    # the frequncey of predictions the ground truth labels for each target
    file_name_proportion_prediction_and_ground_truth_labels =\
    os.path.join(self.DIR_MODEL_PREDICTION_RESULT_THIS_VERSION,\
                 'proportion_prediction_and_ground_truth_labels.csv')
    df_proportion_prediction_and_ground_truth_labels.to_csv(file_name_proportion_prediction_and_ground_truth_labels)


if __name__ == "__main__":
    # reseting the graph is necessary for running the script via spyder or other
    # ipython intepreter
    # pdb.set_trace()

    tf.reset_default_graph()

    preference_predictor = PreferencePredictor()
    # pdb.set_trace()
    # --------------------------------------------------------------
    # parse data and files
    # --------------------------------------------------------------
    preference_predictor.parse_prediction_data_whole_data_set()

    # --------------------------------------------------------------
    # make predictions
    # prediction_proportion = (num_classes, 1)
    # data_set_predicted_labels = (num_files, 1)
    # data_set_ground_truth_labels = (num_files, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    preference_predictor.predict_preferences()

    # pdb.set_trace()

    # --------------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------------
    preference_predictor.save_predictions()
    # pdb.set_trace()





