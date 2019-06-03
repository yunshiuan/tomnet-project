#-Path
# Change working directory to where the script locates
import os
#os.getcwd()
#PATH_ROOT = '/Users/vimchiz/bitbucket_local/observer_model_group/benchmark'
#os.chdir(PATH_ROOT)

import os
import sys
import time
import math
import datetime
import numpy as npc
import pandas as pd
import tensorflow as tf
import commented_resnet as rn
import sys
#sys.path.insert(0, '/temporary_testing_version')
#import data_handler as dh
import commented_data_handler as dh
import argparse
import itertools
import numpy as np
# For debugging
import pdb

class Model:
  HEIGHT = 12 # height of the maze
  WIDTH = 12 # width of the maze
  # DEPTH != MAX_TRAJECTORY_SIZE (see commented_data_handler.py)
  # - MAX_TRAJECTORY_SIZE = 10, number of steps of each trajectory 
  # (will be padded up/truncated to it if less/more than the constant)
    # - DEPTH = number of channels of each maze, 11 = 1 (obstacle) + 4 (targets) + 1 (agent initial position) + 5 (actions)
  MAX_TRAJECTORY_SIZE = 10
  DEPTH = 11 
  
  #Batch size = 16, same in the paper A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  BATCH_SIZE_TRAIN = 16 # size of the batch for traning (number of the steps within each batch)  
  BATCH_SIZE_VAL = 16 # size of the batch for validation
  BATCH_SIZE_TEST = 16 # size of batch for testing
  
  # number of layers in the resnet 
  # (5, same in the paper, A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  NUM_RESIDUAL_BLOCKS = 5
  TRAIN_EMA_DECAY = 0.95
  

  
  # --------------------------------------
#  ## for testing on the local machine with 1000 files and few steps
#  subset_size = 1000
#
#  EPOCH_SIZE = 8000
#  TRAIN_STEPS = 30
#  REPORT_FREQ = 10 # the frequency of writing the error to error.csv
#  # For testing on 1000 files
#  txt_data_path = os.getcwd() + '/test_on_human_data/S030/'
#  
#  ckpt_fname = 'test_on_human_data/training_result/caches/cache_S002a_v??_commit_???_epoch80000_tuning_batch16_train_step_2M_INIT_LR_10-4'
#  train_fname = 'test_on_human_data/training_result/caches/cache_S002a_v??_commit_???_epoch80000_tuning_batch16_train_step_2M_INIT_LR_10-4'

  # --------------------------------------
  
  # --------------------------------------
  # for testing on a GPU machine with 10000 files
  
  # the data size of an epoch (should equal to the traning set size)
  # e.g., given a full date set with 100,000 snapshots,
  # with a train:dev:test = 8:2:2 split,
  # EPOCH_SIZE should be 80,000 training steps if there are 10,000 files
  # because each file contains 10 steps
  
  EPOCH_SIZE = 8000
  subset_size = 1000 # use all files

  # tota number of minibatches used for training
  # (Paper: 2M minibatches, A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  
  TRAIN_STEPS = 50000

  REPORT_FREQ = 100 # the frequency of writing the error to error.csv
  # For testing on 1000 files
  txt_data_path = os.getcwd() + '/S002a/'

  ckpt_fname = 'training_result/caches/cache_S002a_v11_commit_ce0992_epoch80000_tuning_batch16_train_step_0.5M_INIT_LR_10-4'
  train_fname = 'training_result/caches/cache_S002a_v11_commit_ce0992_epoch80000_tuning_batch16_train_step_0.5M_INIT_LR_10-4'
  # --------------------------------------


  # TRUE: use the full data set for validation 
  # (but this would not be fair because a portion of the data has already been seen)
  # FALSE: data split using train:vali:test = 8:1:1
  FULL_VALIDATION = False 

  # Initial learning rate (LR) # paper: 10−4
  INIT_LR = 0.0001  # 10-4
  #DECAY_STEP_0 = 10000 # LR decays for the first time (*0.9) at 10000th steps
  #DECAY_STEP_1 = 15000 # LR decays for the second time (*0.9) at 15000th steps

  NUM_CLASS = 4 # number of unique classes in the training set

  use_ckpt = False
  
  def __init__(self, args):
    '''
    The constructor for the Model class.
    '''
    #The data points must be given one by one here?
    #But the whole trajectory must be given to the LSTM
    
    # placeholder for the trainging traj
    # --------------------------------------------------------------
    # Edwinn's codes
    # self.traj_placeholder, vali_traj_placeholder: (for input_tensor)
    # - shape: 
    # (batch size = 16, width = 12, height = 12, depth = 11)
    # --------------------------------------------------------------
  
#    self.traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TRAIN, self.HEIGHT, self.WIDTH, self.DEPTH])
#    # placeholder for the trainging goal
#    self.goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_TRAIN])
#    # placeholder for the validation traj
#    self.vali_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_VAL, self.HEIGHT, self.WIDTH, self.DEPTH])
#    # placeholder for the validation goal
#    self.vali_goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_VAL])
#    # placeholder for learning rate
#    self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
    
    # --------------------------------------------------------------
    # Paper
    # self.traj_placeholder: (for input_tensor)
    # - shape: 
    # (batch size = 16: batch_size, 10: MAX_TRAJECTORY_SIZE, HEIGHT = 12, WIDTH = 12, DEPTH = 11)
    # --------------------------------------------------------------
    
    ckpt_path = self.ckpt_fname + '/logs/model.ckpt'
    train_path = self.train_fname + '/train/'
    
    self.ckpt_path = ckpt_path
    self.train_path = train_path    
    self.traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TRAIN, self.MAX_TRAJECTORY_SIZE, self.HEIGHT, self.WIDTH, self.DEPTH])
    # placeholder for the trainging goal
    self.goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_TRAIN])
    # placeholder for the validation traj
    self.vali_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_VAL, self.MAX_TRAJECTORY_SIZE, self.HEIGHT, self.WIDTH, self.DEPTH])
    # placeholder for the validation goal
    self.vali_goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_VAL])
#    # placeholder for learning rate
    self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        
    # Load data
    dir = self.txt_data_path
    # pdb.set_trace()
    data_handler = dh.DataHandler(dir)
    # For S002a:
    # Get the data by "data_handler.parse_trajectories(dir, mode=args.mode, shuf=args.shuffle)"
    # self.train_data.shape: (800, 12, 12, 11)
    # self.test_data.shape: (100, 12, 12, 11)
    # self.test_data.shape: (100, 12, 12, 11)
    # self.train_labels.shape: (800, )
    # self.test_labels.shape: (100, )
    # self.test_labels.shape: (100, )
    # len (files) = 100
    # Each data example is one trajectory (each contains 10 steps, MAX_TRAJECTORY_SIZE)
    
    # Note that all training examples are NOT shuffled randomly (by defualt)
    # during data_handler.parse_trajectories()
    
    # pdb.set_trace()
    self.train_data, self.vali_data, self.test_data, self.train_labels, self.vali_labels, self.test_labels, self.all_files, self.train_files, self.vali_files, self.test_files = data_handler.parse_trajectories(dir,mode=args.mode,shuf=args.shuffle, subset_size = self.subset_size)
    # on my local machine: 
    # 'S002_83.txt', 'S002_97.txt', 'S002_68.txt', 'S002_40.txt', 'S002_54.txt', 'S002_55.txt'
    #print('End of __init__-----------------')
    # pdb.set_trace()

    
            
  def _create_graphs(self):
    
    # > for step in range(self.TRAIN_STEPS):
    # The "step" values will be input to 
    # (1)"self.train_operation(global_step, self.full_loss, self.train_top1_error)",
    # and then to
    # (2)"tf.train.ExponentialMovingAverage(self.TRAIN_EMA_DECAY, global_step)"
    # - decay = self.TRAIN_EMA_DECAY 
    # - num_updates = global_step #this is where 'global_step' goes

    global_step = tf.Variable(0, trainable=False)
    validation_step = tf.Variable(0, trainable=False)
    
    # The charnet
    # def build_charnet(input_tensor, n, num_classes, reuse, train):
    # - Add n residual layers
    # - Add average pooling
    # - Add LSTM layer
    # - Add a fully connected layer
    # The output of charnet is "logits", which will be feeded into 
    # the softmax layer to make predictions
    
    # "logits" is the output of the charnet (including ResNET and LSTM) 
    # and is the input for a softmax layer (see below)
    logits = rn.build_charnet(self.traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=False, train=True)
    vali_logits = rn.build_charnet(self.vali_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=True)
    
    # REGULARIZATION_LOSSES: regularization losses collected during graph construction.
    # See: https://www.tensorflow.org/api_docs/python/tf/GraphKeys
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # Training loss and error
    #  loss: the cross entropy loss given logits and true labels
    #  > loss(logits, labels)
    # Note:
    # (1) To compute loss, it is important to use the output from NN before entering the softmax function
    # https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    # WARNING: This op expects unscaled logits, 
    # since it performs a softmax on logits internally for efficiency. 
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    loss = self.loss(logits, self.goal_placeholder)
    
    #  tf.add_n: Adds all input tensors element-wise.
    #  - Using sum or + might create many tensors in graph to store intermediate result.
    self.full_loss = tf.add_n([loss] + regu_losses) 
    predictions = tf.nn.softmax(logits)

    # Performace metric: prediction error
    # - Note that, by comparison,  the loss function 'def loss(self, logits, labels):'
    # - use the cross entropy loss.

    self.train_top1_error = self.top_k_error(predictions, self.goal_placeholder, 1)

    #Validation loss and error
    self.vali_loss = self.loss(vali_logits, self.vali_goal_placeholder)
    vali_predictions = tf.nn.softmax(vali_logits)
    self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_goal_placeholder, 1)

    #Define operations
    self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss, self.train_top1_error)
    self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)
    
    return
        
  def train(self):
    
    print('Start training-----------------')
    #pdb.set_trace()
    
    #Build graphs
    self._create_graphs()
    

    # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
    # summarizing operations by running summary_op. Initialize a new session
    saver = tf.train.Saver(tf.global_variables()) # <class 'tensorflow.python.training.saver.Saver'>
    summary_op = tf.summary.merge_all() # <class 'tensorflow.python.framework.ops.Tensor'>
    
    # initialize_all_variables (from tensorflow.python.ops.variables) 
    # is deprecated and will be removed after 2017-03-02.
    # Instructions for updating:
    # Use `tf.global_variables_initializer` instead.
    
    init = tf.initialize_all_variables() # <class 'tensorflow.python.framework.ops.Operation'>
    # -----------------------
    # Session: This is the start of the tf session
    # -----------------------
    sess = tf.Session()

    # If you want to load from a checkpoint
    if self.use_ckpt:
      saver.restore(sess, self.ckpt_path)
      print('Restored from checkpoint...')
    else:
      # -----------------------
      # Session: Initialize all the parameters in the sess.
      # See above: "init = tf.initialize_all_variables()"
      # -----------------------
      sess.run(init)
      
    # This summary writer object helps write summaries on tensorboard
    # this is irrelevant to the error.csv file
    summary_writer = tf.summary.FileWriter(self.train_path, sess.graph)

    # These lists are used to save a csv file at last
    # This is the data for error.csv
    step_list = []
    train_error_list = []
    val_error_list = []
        
    print('Start training...')
    print('----------------------------')
    #pdb.set_trace()
    
    for step in range(self.TRAIN_STEPS):
      # pdb.set_trace()

      #Generate batches for training and validation
      # Each example in a batch is of the shape 
      # (maze width = 12, maze height = 12, steps of each trajectory = 11)
      train_batch_data, train_batch_labels = self.generate_train_batch(self.train_data, self.train_labels, self.BATCH_SIZE_TRAIN)
      validation_batch_data, validation_batch_labels = self.generate_vali_batch(self.vali_data, self.vali_labels, self.BATCH_SIZE_VAL)

      #Validate first?
      if step % self.REPORT_FREQ == 0:
        if self.FULL_VALIDATION:
          validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss, top1_error=self.vali_top1_error, vali_data=vali_data, vali_labels=vali_labels, session=sess, batch_data=train_batch_data, batch_label=train_batch_labels)

          vali_summ = tf.Summary()
          vali_summ.value.add(tag='full_validation_error', simple_value=validation_error_value.astype(np.float))
          summary_writer.add_summary(vali_summ, step)
          summary_writer.flush()
        
        else:
          _, validation_error_value, validation_loss_value = sess.run([self.val_op, self.vali_top1_error, self.vali_loss], {self.traj_placeholder: train_batch_data, self.goal_placeholder: train_batch_labels, self.vali_traj_placeholder: validation_batch_data, self.vali_goal_placeholder: validation_batch_labels, self.lr_placeholder: self.INIT_LR})
        
        val_error_list.append(validation_error_value)
      
      start_time = time.time()

      # Actual training
      # -----------------------------------------------
      # This is where the train_error_value comes from 
      # -----------------------------------------------
      # sess.run(
      #     fetches = [self.train_op,
      #                self.train_ema_op,
      #                self.full_loss,
      #                self.train_top1_error], 
      #     feed_dict = {self.traj_placeholder: train_batch_data,
      #                  self.goal_placeholder: train_batch_labels,
      #                  self.vali_traj_placeholder: validation_batch_data,
      #                  self.vali_goal_placeholder: validation_batch_labels,
      #                  self.lr_placeholder: self.INIT_LR})
      # Parameters:
      # -----------------------------
      # fetches
      # -----------------------------
      # (1,2) self.train_op, self.train_ema_op 
      # - (1) These define the optimization operation.
      # - (2) come from: def _create_graphs(self):
      #       (1) come from: self.train_operation(global_step, self.full_loss, self.train_top1_error)
      #         - return: two operations. 
      #           - Running train_op will do optimization once. 
      #           - Running train_ema_op will generate the moving average of train error and 
      #             train loss for tensorboard
      #         - param: global_step #TODO ??
      #         - param: self.full_loss: 
      #             - The loss that includes both the loss and the regularized loss
      #             - comes from: self.full_loss = tf.add_n([loss] + regu_losses)
      #         - param: self.train_top1_error: 
      #             def _create_graphs(self):
      #                self.train_top1_error = self.top_k_error(predictions, self.goal_placeholder, 1)
      #                   def top_k_error(self, predictions, labels, k):
      #                        The Top-1 error is the percentage of the time that the classifier 
      #                        did not give the correct class the highest score.
      #
      # (3) self.full_loss
      # - (1) The loss that includes both the loss and the regularized loss
      # - (2) comes from: self.full_loss = tf.add_n([loss] + regu_losses)
      #
      # (4) self.train_top1_error
      # - (1) comes from:
      # - def _create_graphs(self):
      # --- self.train_top1_error = self.top_k_error(predictions, self.goal_placeholder, 1)
      # --- def top_k_error(self, predictions, labels, k):
      # - (2) The Top-1 error is the percentage of the time that the classifier 
      #       did not give the correct class the highest score.
      #
      # -----------------------------
      # feed_dict
      # -----------------------------
      # self.traj_placeholder: train_batch_data
      # - feed in the trajectories of the training batch
      # self.goal_placeholder: train_batch_labels
      # - feed in the labels of the training batch
      # self.vali_traj_placeholder: validation_batch_data
      # - feed in the trajectories of the validation batch
      # self.vali_goal_placeholder: validation_batch_labels
      # - feed in the labels of the validation batch
      # self.lr_placeholder: self.INIT_LR
      # - feed in the initial learning rate
      
      _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op, self.full_loss, self.train_top1_error], {self.traj_placeholder: train_batch_data, self.goal_placeholder: train_batch_labels, self.vali_traj_placeholder: validation_batch_data, self.vali_goal_placeholder: validation_batch_labels, self.lr_placeholder: self.INIT_LR})
      duration = time.time() - start_time

      if step % self.REPORT_FREQ == 0:
        summary_str = sess.run(summary_op, {self.traj_placeholder: train_batch_data, self.goal_placeholder: train_batch_labels, self.vali_traj_placeholder: validation_batch_data, self.vali_goal_placeholder: validation_batch_labels, self.lr_placeholder: self.INIT_LR})
        summary_writer.add_summary(summary_str, step)

        num_examples_per_step = self.BATCH_SIZE_TRAIN # trajectoris per step = trajectoris per batch = batch size
        examples_per_sec = num_examples_per_step / duration # trajectories per second
        sec_per_batch = float(duration) # seconds for this step

        format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
        print(format_str % (datetime.datetime.now(), step, train_loss_value, examples_per_sec, sec_per_batch))
        print('Train top1 error = ', train_error_value)
        print('Validation top1 error = %.4f' % validation_error_value)
        print('Validation loss = ', validation_loss_value)
        print('----------------------------')

        # This records the training steps and the corresponding training error
        step_list.append(step)
        train_error_list.append(train_error_value)
        
        #print('End of training report-----------------')
        #pdb.set_trace()
            
      #if step == self.DECAY_STEP_0 or step == self.DECAY_STEP_1:
      #  self.INIT_LR = 0.1 * self.INIT_LR
      #  print('Learning rate decayed to ', self.INIT_LR)
        
      # Save checkpoints every 10000 steps and also at the last step      
      if step % 10000 == 0 or (step + 1) == self.TRAIN_STEPS:
          checkpoint_path = os.path.join(self.train_path, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

          df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                  'validation_error': val_error_list})
          # overwrite the csv
          df.to_csv(self.train_path + '_error.csv')

  def test(self):
    '''
    This function is used to evaluate the validation and test data. Please finish pre-precessing in advance
    It will write a csv file with both validation and test perforance.
    '''

    # --------------------------------------------------------------
    # Evaluate the model on the whole validation set
    # --------------------------------------------------------------
    # pdb.set_trace()
    df_vali_all = self.evaluate_on_validation_set()

    # --------------------------------------------------------------
    # Evaluate the model on the whole test set
    # --------------------------------------------------------------
    # pdb.set_trace()
    df_test_all = self.evaluate_on_test_set()
    
    # --------------------------------------------------------------
    # My codes
    # Combine all dfs into one
    # -------------------------------------------------------------- 
    #pdb.set_trace()

    df_vali_and_test = df_vali_all.append(df_test_all)
    
    df_vali_and_test.to_csv(self.train_path + '_test_and_validation_accuracy.csv')

    return df_vali_and_test

  def evaluate_on_test_set(self):
      '''
      Evaluate a model with the test data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.
      
      Returns:
        :df_accuracy_all: a dataframe with model accuracy.
      '''
  
      df_accuracy_all = self.evaluate_whole_data_set(self.test_files, self.test_data, self.test_labels, self.BATCH_SIZE_TEST, 'test')
      
      return df_accuracy_all
    
  def evaluate_on_validation_set(self):
      '''
      Evaluate a model with the validation data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.
      
      Returns:
        :df_accuracy_all: a dataframe with model accuracy.
      '''
  
      df_accuracy_all = self.evaluate_whole_data_set(self.vali_files, self.vali_data, self.vali_labels, self.BATCH_SIZE_VAL, 'vali')
      
      return df_accuracy_all
    
  def evaluate_whole_data_set(self, files, data, labels, batch_size, mode):
      '''
      Evaluate a model with a set of data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.
      
      Args:
        :param files: the txt files to be test (only used to compute the number of trajectories)
        :param data: the data to be test the model on (num_files * MAX_TRAJECTORY_SIZE, height, width, depth)
        :param labels: the ground truth labels to be test the model on (num_files * MAX_TRAJECTORY_SIZE, 1)
        :param batch_size: the batch size
        :param mode: should be either 'vali' or 'test'
        
      Returns:
        :df_accuracy_all: a dataframe with model accuracy.
      '''
      # pdb.set_trace()
     
      num_vali_files = len(files)
      num_batches = num_vali_files // batch_size
      # remain_trajs = num_vali_steps % self.BATCH_SIZE_TEST
      print('%i validation batches in total...' %num_batches)
  
      # Create the image and labels placeholders
      traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_VAL, self.MAX_TRAJECTORY_SIZE, self.HEIGHT, self.WIDTH, self.DEPTH])
  
      # Build the vali graph
      logits = rn.build_charnet(traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=False)
      # logits = (batch_size, num_classes)
      predictions = tf.nn.softmax(logits)
      # predictions = (batch_size, num_classes)
  
      # Initialize a new session and restore a checkpoint
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
        file_index = step * self.BATCH_SIZE_VAL
        batch_data, batch_labels = self.generate_vali_batch(data, labels, batch_size, file_index)
        # pdb.set_trace()
        batch_prediction_array = sess.run(predictions, feed_dict={traj_placeholder: batch_data})
        # batch_prediction_array = (batch_size, num_classes)
        data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
        # vali_set_prediction_array will be size of (batch_size * num_batches, num_classes)
        data_set_ground_truth_labels = np.concatenate((data_set_ground_truth_labels, batch_labels))
      # --------------------------------------------------------------
      # Edwinn's codes
      # Test accuracy by match_estimation()
      # Only work for data format (batch, ...)
      # Turned off for data format (batch, timesteps, ...)
      # --------------------------------------------------------------
#      # vali_set_prediction_array = (batch_size * num_batches) x num_classes
#      # length = (batch_size * num_batches)
#      rounded_array = np.around(vali_set_prediction_array,2).tolist()
#      length = num_batches*self.BATCH_SIZE_TEST  
#      df_vali_match_estimation = self.match_estimation(rounded_array, self.vali_labels, length, 'vali')
#      
      # --------------------------------------------------------------
      # My codes
      # Test accuracy by definition
      # --------------------------------------------------------------
      # pdb.set_trace()
      # vali_set_prediction_array = (num_batches*batch_size, num_classes)
      # vali_set_ground_truth = (num_batches*batch_size, 1)
      
      df_accuracy_proportion = self.proportion_accuracy(data_set_prediction_array, data_set_ground_truth_labels, mode)
  
      # --------------------------------------------------------------
      # My codes
      # Combine all dfs into one
      # -------------------------------------------------------------- 
      #pdb.set_trace()
  
#      df_vali_all = df_vali_proportion.append(df_vali_match_estimation, ignore_index = True) 
      df_accuracy_all = df_accuracy_proportion
  
      return df_accuracy_all
    
#  def match_estimation(self,predictions, labels, length, mode):
#    '''
#    [Deprecated! Use proportion_accuracy() instead.
    # Only work for data format (batch, ...)
    # Turned off for data format (batch, timesteps, ...)]
    
#    Evaluate model accuracy defined by Edwinn's method.
#    Return a df that contains the accuracy metric.
#    
#    :param labels: ground truth labels (including both in-batch and out-of-batch
#    labels. Note that only in-batch labels (size = length) are tested because 
#    they have corresponding predicted labels.
#    :param predicitons: predicted labels (num_batches * batch_size, num_classes).
#    :param length: (num_batches * batch_size, 1). This defines the number of 
#    labels that are in batches. Note that some remaining labels are not
#    included in batches.
#    :param mode: should be either 'vali' or 'test'
#    :return df_summary: a a dataframe that stores the acuuracy metrics
#    '''
#    
#    #Initialize zeroes for each possible arrangement
#    matches = [0 for item in range(math.factorial(self.NUM_CLASS))]
#    
#    for i in range(length):
#      #Initialize a 2d zeroes array
#      test = [[0 for item in range(self.NUM_CLASS)] for item in range(math.factorial(self.NUM_CLASS))]
#      combinations = list(itertools.permutations(range(self.NUM_CLASS),self.NUM_CLASS))
#      for j in range(math.factorial(self.NUM_CLASS)):
#        for k in range(self.NUM_CLASS):
#          test[j][k] = predictions[i][combinations[j][k]]
#      
#      for j in range(math.factorial(self.NUM_CLASS)):
#        if int(labels[i]) == test[j].index(max(test[j])):
#          matches[j] += 1
#
#    best = matches.index(max(matches))
#    print('\n' + str(mode) +': match_estimation()')
#    print( 'Combination with best matches was ' + str(combinations[best]))
#    print('Matches: ' + str(matches[best]) + '/' + str(length))
#    print('Accuracy: ' + str(round(matches[best]*100/length,2)) + '%')
#    df_summary = pd.DataFrame(data={'matches':str(str(matches[best]) + '/' + str(length)),
#                                    'accurary':str(str(round(matches[best]*100/length,2)) + '%'),
#                                    'mode': str(mode) + '_match_estimation'},
#                      index = [0])
#    ## write the csv
#    #df.to_csv(self.train_path + '_test_accuracy_v1.csv')
#    return df_summary  
  
  def proportion_accuracy(self, prediction_array, labels, mode):
    '''
    Evaluate model accuracy defined by proportion (num_matches/num_total).
    Return a df that contains the accuracy metric.
    
    Args:
      :param prediction_array: a tensor with (num_batches * batch_size, num_classes).
      :param labels: in-batch labels. Note that only in-batch labels (size = length) 
        are tested because they have corresponding predicted labels.
      :param mode: should be either 'vali' or 'test'
    Returns:
      :df_summary: a a dataframe that stores the acuuracy metrics
    '''
    total_predictions = len(prediction_array)
    # match_predictions
    predicted_labels = np.argmax(prediction_array,1)
    
    # Retrieve corresponding labels
    groud_truth_labels = labels.astype(int)
    # pdb.set_trace()
    match_predictions = sum(predicted_labels == groud_truth_labels)

    matches_percentage = str(match_predictions) + '/' + str(total_predictions)
    accuracy = str(round(match_predictions*100/total_predictions, 2)) + '%'
    
    print('\n' + str(mode)+ ': proportion_accuracy()')
    print('Matches: ' + matches_percentage)
    print('Accuracy: ' + accuracy)
    
    df_summary = pd.DataFrame(data={'matches':matches_percentage,
                                    'accurary':accuracy,
                                    'mode': str(mode + '_proportion')},
                        index = [0])
    return df_summary
  
  def loss(self, logits, labels):
    '''
    Calculate the cross entropy loss given logits and true labels
    :param logits: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size]
    :return: loss tensor with shape [1]
    '''
    labels = tf.cast(labels, tf.int64)
    
    # Note
    # (1) https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    # WARNING: This op expects unscaled logits, 
    # since it performs a softmax on logits internally for efficiency. 
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    # (2) The ToMNET paper also uses softmax cross entropy for loss function
    # https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-softmax-crossentropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

  def top_k_error(self, predictions, labels, k):
    '''
    Calculate the top-k error
    :param predictions: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size, 1]
    :param k: int
    :return: tensor with shape [1]
    '''
    # -----------
    # The Top-1 error is the percentage of the time that the classifier 
    # did not give the correct class the highest score. The Top-5 error 
    # is the percentage of the time that the classifier did not include 
    # the correct class among its top 5 guesses.
    # -----------

    # predictions:
    # Tensor("Softmax_1:0", shape=(16, 4), dtype=float32)
    batch_size = predictions.get_shape().as_list()[0]
    
    # in_top1
    # Tensor("ToFloat_1:0", shape=(16,), dtype=float32)
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    
    # num_correct
    # Tensor("Sum_1:0", shape=(), dtype=float32)
    num_correct = tf.reduce_sum(in_top1)
    # print('predictions:')
    # print(predictions)
    # print('in_top1')
    # print(in_top1) 
    # print('num_correct')
    # print(num_correct)
    return (batch_size - num_correct) / float(batch_size)
  
  def train_operation(self, global_step, total_loss, top1_error):
    '''
    Defines train operations
    
    :param global_step: tensor variable with shape [1]
    :param total_loss: tensor with shape [1]
    :param top1_error: tensor with shape [1]
    :return: two operations. Running train_op will do optimization once. Running train_ema_op
    will generate the moving average of train error and train loss for tensorboard
    '''
    # Add train_loss, current learning rate and train error into the tensorboard summary ops
    tf.summary.scalar('learning_rate', self.lr_placeholder)
    tf.summary.scalar('train_loss', total_loss)
    tf.summary.scalar('train_top1_error', top1_error)

    # The ema object help calculate the moving average of train loss and train error
    ema = tf.train.ExponentialMovingAverage(self.TRAIN_EMA_DECAY, global_step)
    train_ema_op = ema.apply([total_loss, top1_error])
    tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
    tf.summary.scalar('train_loss_avg', ema.average(total_loss))

    opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)
    train_op = opt.minimize(total_loss)
    return train_op, train_ema_op

  def validation_op(self, validation_step, top1_error, loss):
    '''
    Defines validation operations
    
    :param validation_step: tensor with shape [1]
    :param top1_error: tensor with shape [1]
    :param loss: tensor with shape [1]
    :return: validation operation
    '''

    # This ema object help calculate the moving average of validation loss and error

    # ema with decay = 0.0 won't average things at all. This returns the original error
    ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
    ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


    val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]), ema2.apply([top1_error, loss]))
    top1_error_val = ema.average(top1_error)
    top1_error_avg = ema2.average(top1_error)
    loss_val = ema.average(loss)
    loss_val_avg = ema2.average(loss)

    # Summarize these values on tensorboard
    tf.summary.scalar('val_top1_error', top1_error_val)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)
    tf.summary.scalar('val_loss', loss_val)
    tf.summary.scalar('val_loss_avg', loss_val_avg)
    
    return val_op
  
  def generate_train_batch(self, train_data, train_labels, train_batch_size, file_index = -1):
    '''
    This function helps you generate a batch of training data.
    
    Args:
      :param train_data: 4D numpy array (total_steps, height, width, depth)
      :param train_labels: 1D numpy array (total_steps, ）
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1, a random batch will be chosen from the data set.
  
    Returns: 
      :train_batch_data:
        a batch data. 4D numpy array (batch_size, trajectory_size, height, width, depth) and 
      :train_batch_labels: a batch of labels. 1D numpy array (batch_size, )
    '''
    # geneate random batches
    train_batch_data, train_batch_labels = self.generate_batch(train_data, train_labels, train_batch_size, file_index)
      
    return train_batch_data, train_batch_labels
  
  def generate_vali_batch(self, vali_data, vali_labels, vali_batch_size, file_index = -1):
    '''
    This function helps you generate a batch of validation data.
    
    Args:
      :param vali_data: 4D numpy array (total_steps, height, width, depth)
      :param vali_labels: 1D numpy array (total_steps, ）
      :param vali_batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1 (default),
        a random batch will be chosen from the data set.
  
    Returns: 
      :vali_batch_data:
        a batch data. 4D numpy array (batch_size, trajectory_size, height, width, depth) and 
      :vali_batch_labels: a batch of labels. 1D numpy array (batch_size, )
    '''
    # geneate random batches
    vali_batch_data, vali_batch_labels = self.generate_batch(vali_data, vali_labels, vali_batch_size, file_index)
      
    return vali_batch_data, vali_batch_labels

  def generate_test_batch(self, test_data, test_labels, test_batch_size, file_index):
    '''
    This function helps you generate a batch of test data.
    
    Args:
      :param test_data: 4D numpy array (total_steps, height, width, depth)
      :param test_labels: 1D numpy array (total_steps, ）
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1, a random batch will be chosen from the data set.
  
    Returns: 
      :test_batch_data:
        a batch data. 4D numpy array (batch_size, trajectory_size, height, width, depth) and 
      :test_batch_labels: a batch of labels. 1D numpy array (batch_size, )
    '''

    test_batch_data, test_batch_labels = self.generate_batch(test_data, test_labels, test_batch_size, file_index)
    return test_batch_data, test_batch_labels
  
  def generate_batch(self, data, labels, batch_size, file_index):
    '''
    This function helps you generate a batch of data.
    
    Args:
      :param data: 4D numpy array (total_steps, height, width, depth)
      :param labels: 1D numpy array (total_steps, ）
      :param batch_size: int
      :param file_index: the starting index of the batch in the data set. 
        If set to the special number -1, a random batch will be chosen from the data set.

    Returns: 
      :batch_data:
        a batch data. 4D numpy array (batch_size, trajectory_size, height, width, depth) and 
      :batch_labels: a batch of labels. 1D numpy array (batch_size, )
    '''
    # --------------------------------------------------------------
    # Edwinn's codes
    # Generate a batch. 
    # Each batch is a step.
    # Each batch contains 16 steps (span from 2 trajectories, which
    # each contains 10 steps).
    # batch_data shape = (16, 6, 6, 11)
    # batch_label shape = (16, 1)
    # --------------------------------------------------------------
#    offset = np.random.choice(self.EPOCH_SIZE - test_batch_size, 1)[0]
#    test_data_batch = test_data[offset:offset+test_batch_size, ...]
#    vali_label_batch = vali_label[offset:offset+test_batch_size]
    
    # --------------------------------------------------------------
    # Paper codes
    # Generate a batch. 
    # Each example is a trejectory.
    # Each batch contains 16 examples (trajectories). Each trajectory contains 10 steps.
    # batch_data shape = (16, 10, 12, 12, 11)
    # batch_label shape = (16, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
        
    # the total number of batch equals the total number of steps devided by the steps for each trajectory
    # (e.g., # training steps = 8000, max_trajectory_size = 10, then total_number_file = 800)
    num_files = int(np.ceil(len(data)/self.MAX_TRAJECTORY_SIZE)) 

    # --------------------------------------------------------------
    # Reshape train_data
  
    # train_data = (num_steps, height, width, depth) ->
    # train_data = (num_files, num_steps, height, width, depth)
    # --------------------------------------------------------------
    
    # train_data = (num_steps, height, width, depth)
    data = data.reshape((num_files, self.MAX_TRAJECTORY_SIZE,
                         self.HEIGHT, self.WIDTH, self.DEPTH))
    # train_data = (num_files, num_steps, height, width, depth)

    # --------------------------------------------------------------
    # Chose train_batch_size random files from all the files
    # test_data = (num_files, num_steps, height, width, depth)->
    # batch_data = (batch_size, num_steps, height, width, depth)->
    # --------------------------------------------------------------
    if file_index == -1:
      # choose a random set of files (could be not continuous)
      indexes_files = np.random.choice(num_files, batch_size)
    else:
      # choose a continuous series of files 
      indexes_files = range(file_index, file_index + batch_size)
      
    batch_data = data[indexes_files,...]

    # --------------------------------------------------------------
    # Only retain unique labels
    # test_labels = （total_steps, ） ->
    # test_labels = (num_files, )
    # --------------------------------------------------------------
    # test_labels = (1000,)    
    labels = labels[0:-1:self.MAX_TRAJECTORY_SIZE]
    # test_labels = (100, ) 
    
    # --------------------------------------------------------------    
    # Choose the labels coresponding to the indexes files
    # test_labels = （num_files, ） ->
    # batch_labels = (batch_size, )
    # (100,) -> (16,)
    # --------------------------------------------------------------   
    batch_labels = labels[indexes_files]
    # pdb.set_trace()
    assert(batch_labels.shape[0]==batch_size)
    
    return batch_data, batch_labels



if __name__ == "__main__":
    # reseting the graph is necessary for running the script via spyder or other
    # ipython intepreter
    tf.reset_default_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', help='all: train and test, train: only train, test: only test')
    parser.add_argument('--shuffle', type=str, default=False, help='shuffle the data for more random result')
    
    args = parser.parse_args()	
    model = Model(args)
    
    if args.mode == 'train' or args.mode == 'all':
      model.train()
    if args.mode == 'test' or args.mode == 'all':
      model.test()
    
