#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class Model(mp.ModelParameter):

The class for training the ToMNET model.

Note:
  Inherit mp.ModelParameter to share model constants.
  
@author: Chuang, Yun-Shiuan; Edwinn
"""
import os
import sys
import time
import datetime
import pandas as pd
import tensorflow as tf
import sys
#sys.path.insert(0, '/temporary_testing_version')
#import data_handler as dh
import argparse
import numpy as np
# For debugging
import pdb
import commented_charnet as cn
import commented_prednet as pn
import commented_data_handler as dh
import commented_model_parameters as mp
import commented_batch_generator as bg

class Model(mp.ModelParameter):
  # --------------------------------------
  # Constant block
  # --------------------------------------
  # --------------------------------------
  # Constant: Model parameters
  # --------------------------------------
  # Use inheretance to share the model constants across classes
  
  # --------------------------------------
  # Constant: Training parameters
  # --------------------------------------
  #Batch size = 16, same in the paper A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  BATCH_SIZE_TRAIN = 5 # size of the batch for traning (number of the steps within each batch)  
  BATCH_SIZE_VAL = 5 # size of the batch for validation
  BATCH_SIZE_TEST = 5 # size of batch for testing
  
  # for testing on a GPU machine with 10000 files  
  SUBSET_SIZE = 100 # use all files
  # tota number of minibatches used for training
  # (Paper: 2M minibatches, A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  TRAIN_STEPS = 50
  REPORT_FREQ = 10 # the frequency of writing the error to error.csv
  #txt_data_path = os.getcwd() + '/S002a/'
  # TRUE: use the full data set for validation 
  # (but this would not be fair because a portion of the data has already been seen)
  # FALSE: data split using train:vali:test = 8:1:1
  FULL_VALIDATION = False 
  USE_CKPT = False
  
  # --------------------------------------
  # Variable: Training parameters
  # --------------------------------------  
  path_mode =  os.getcwd()  # Necessary when the output dir and script dir is different
  ckpt_fname = 'training_result/caches/cache_S030_v16_commit_???_epoch80000_tuning_batch96_train_step_1K_INIT_LR_10-4'
  train_fname = 'training_result/caches/cache_S030_v16_commit_???_epoch80000_tuning_batch96_train_step_1K_INIT_LR_10-4'
  txt_data_path ='../../data/S002a/'
  #txt_data_path = os.getcwd() + '/test_on_human_data/data/processed/S030/'
  ckpt_fname = os.path.join(path_mode,ckpt_fname)
  train_fname = os.path.join(path_mode,train_fname)
  txt_data_path = os.path.join(path_mode,txt_data_path)
  
  def __init__(self, args):
    '''
    The constructor for the Model class.
    ''' 
    # --------------------------------------------------------------
    # Set up constants
    # --------------------------------------------------------------
    ckpt_path = self.ckpt_fname + '/logs/model.ckpt'
    train_path = self.train_fname + '/train/'
    
    self.ckpt_path = ckpt_path
    self.train_path = train_path  
    
    # Set up batch generator
    self.batch_generator = bg.BatchGenerator()
  
    # --------------------------------------------------------------
    # Set up all the placeholders
    # --------------------------------------------------------------
    self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    # For trajectory data and the corresponding labels 
    self.train_data_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TRAIN, self.MAX_TRAJECTORY_SIZE, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_TRAJECTORY])
    self.train_labels_traj_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_TRAIN])
    self.vali_data_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_VAL, self.MAX_TRAJECTORY_SIZE, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_TRAJECTORY])
    self.vali_labels_traj_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_VAL])
        
    # For query state data and the cooresponding labels
    self.train_data_query_state_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TRAIN, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_QUERY_STATE])
    self.train_labels_query_state_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_TRAIN])
    self.vali_data_query_state_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_VAL, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_QUERY_STATE])
    self.vali_labels_query_state_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_VAL])

    # --------------------------------------------------------------
    # Parse the trajectory data and labels
    # train_data_traj = (num_train_files, trajectory_size, height, width, MAZE_DEPTH)
    # train_labels_traj = (num_train_files, )
    # --------------------------------------------------------------
    # Load data
    dir = self.txt_data_path
    # pdb.set_trace()
    data_handler = dh.DataHandler()
    # Note that all training examples are NOT shuffled randomly (by defualt)
    # during data_handler.parse_trajectories()
    # pdb.set_trace()
    self.train_data_traj, self.vali_data_traj,\
    self.test_data_traj, self.train_labels_traj,\
    self.vali_labels_traj, self.test_labels_traj,\
    self.all_files_traj, self.train_files_traj, self.vali_files_traj, self.test_files_traj\
    = data_handler.parse_whole_data_set(dir,\
                                        mode=args.mode,\
                                        shuf=args.shuffle,\
                                        subset_size = self.SUBSET_SIZE,\
                                        parse_query_state = False)
    # --------------------------------------------------------------
    # Parse the query state data and labels
    # train_data_traj = (num_train_files, height, width, MAZE_DEPTH_QUERY_STATE)
    # train_labels_traj = (num_train_files, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    self.train_data_query_state, self.vali_data_query_state,\
    self.test_data_query_state, self.train_labels_query_state,\
    self.vali_labels_query_state, self.test_labels_query_state,\
    self.all_files_query_state, self.train_files_query_state, self.vali_files_query_state, self.test_files_query_state \
    = data_handler.parse_whole_data_set(dir,\
                                        mode=args.mode,\
                                        shuf=args.shuffle,\
                                        subset_size = self.SUBSET_SIZE,\
                                        parse_query_state = True)                                                                                                                                                                                                                                             

    #print('End of __init__-----------------')
    #pdb.set_trace()
            
  def _create_graphs(self, with_prednet):
    '''
    Create the graph that includes all tensforflow operations and parameters.
    
    Args:
      :with_prednet:
        If `with_prednet = True`, then construct the complete model includeing
        both charnet and prednet.
        If `with_prednet = False`, then construct the partial model including
        only the charnet.

    '''
       
    # > for step in range(self.TRAIN_STEPS):
    # The "step" values will be input to 
    # (1)"self.train_operation(global_step, self.full_loss, self.train_top1_error)",
    # and then to
    # (2)"tf.train.ExponentialMovingAverage(self.TRAIN_EMA_DECAY, global_step)"
    # - decay = self.TRAIN_EMA_DECAY 
    # - num_updates = global_step #this is where 'global_step' goes

    global_step = tf.Variable(0, trainable=False)
    validation_step = tf.Variable(0, trainable=False)
    
    #pdb.set_trace()
    
    # --------------------------------------------------------------
    # Build the model for training and validation
    # --------------------------------------------------------------
    if not with_prednet:
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
      charnet = cn.CharNet()

      logits = charnet.build_charnet(self.train_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=False, train=True)
      # - Use train=True for batch-wise validation along training to make the error metric
      # - comparable to training error
      vali_logits = charnet.build_charnet(self.vali_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=True)
    else:
      charnet = cn.CharNet()
      prednet = pn.PredNet()
      length_e_char = length_e_char = self.LENGTH_E_CHAR

      
      # model for training
      # pdb.set_trace()
      train_e_char = charnet.build_charnet(self.train_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=length_e_char, reuse=False, train=True)      
      logits = prednet.build_prednet(train_e_char, self.train_data_query_state_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes = self.NUM_CLASS, reuse=False )
      
      # model for batch-validation along training
      # - Use train=True for batch-wise validation along training to make the error metric
      # - comparable to training error
      vali_e_char = charnet.build_charnet(self.vali_data_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=length_e_char, reuse=True, train=True)      
      vali_logits = prednet.build_prednet(vali_e_char, self.vali_data_query_state_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes = self.NUM_CLASS, reuse=True )
      
    # --------------------------------------------------------------
    # Define the regularization operation for training
    # --------------------------------------------------------------
    # REGULARIZATION_LOSSES: regularization losses collected during graph construction.
    # See: https://www.tensorflow.org/api_docs/python/tf/GraphKeys
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # --------------------------------------------------------------
    # Define training loss and error
    # Note that for training, regularization is required;
    # however, for validation, regularization is not needed.
    # --------------------------------------------------------------  
    #  loss: the cross entropy loss given logits and true labels
    #  > loss(logits, labels)
    # Note:
    # (1) To compute loss, it is important to use the output from NN before entering the softmax function
    # https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    # WARNING: This op expects unscaled logits, 
    # since it performs a softmax on logits internally for efficiency. 
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    loss = self.loss(logits, self.train_labels_traj_placeholder)
    
    #  tf.add_n: Adds all input tensors element-wise.
    #  - Using sum or + might create many tensors in graph to store intermediate result.
    self.full_loss = tf.add_n([loss] + regu_losses) 
    
    #Validation loss and error
    self.vali_loss = self.loss(vali_logits, self.vali_labels_traj_placeholder)

    # --------------------------------------------------------------
    # Make prediction based on the output of the model
    # --------------------------------------------------------------  
    predictions = tf.nn.softmax(logits, name = 'train_prediction')
    vali_predictions = tf.nn.softmax(vali_logits, name = 'vali_prediction')

    # --------------------------------------------------------------
    # Define performace metric: prediction error
    # - Note that, by comparison,  the loss function 'def loss(self, logits, labels):'
    # - use the cross entropy loss.
    # --------------------------------------------------------------
    self.train_top1_error = self.top_k_error(predictions, self.train_labels_traj_placeholder, 1)
    self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_labels_traj_placeholder, 1)

    # --------------------------------------------------------------
    # Define optimizer
    # --------------------------------------------------------------
    self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss, self.train_top1_error)
    self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)
    
    return
        
  def train(self):
    
    print('Start training-----------------')
    # pdb.set_trace()
    
    #Build graphs
    self._create_graphs(with_prednet = self.WITH_PREDNET)
    

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
    if self.USE_CKPT:
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
        
    print('Start training batch by batch...')
    print('----------------------------')
    #pdb.set_trace()
    
    for step in range(self.TRAIN_STEPS):
      # pdb.set_trace()
      # --------------------------------------------------------------
      # Generate batches for training data
      # --------------------------------------------------------------
      #pdb.set_trace()
      train_batch_data_traj, train_batch_labels_traj,\
      train_batch_data_query_state, train_batch_labels_query_state\
      = self.batch_generator.generate_train_batch(self.train_data_traj,\
                                  self.train_labels_traj,\
                                  self.train_data_query_state,\
                                  self.train_labels_query_state,\
                                  self.BATCH_SIZE_TRAIN)

      # --------------------------------------------------------------
      # Generate batches for validation data
      # --------------------------------------------------------------
      vali_batch_data_traj, vali_batch_labels_traj,\
      vali_batch_data_query_state, vali_batch_labels_query_state\
      = self.batch_generator.generate_vali_batch(self.vali_data_traj,\
                                  self.vali_labels_traj,\
                                  self.vali_data_query_state,\
                                  self.vali_labels_query_state,\
                                  self.BATCH_SIZE_TRAIN)

      #Validate first?
      if step % self.REPORT_FREQ == 0:

        # Comment the block for 'FULL_VALIDATION' as it will not be run anyways
#        if self.FULL_VALIDATION:
#          validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss, top1_error=self.vali_top1_error, vali_data=vali_data, vali_labels=vali_labels, session=sess, batch_data=train_batch_data, batch_label=train_batch_labels)
#
#          vali_summ = tf.Summary()
#          vali_summ.value.add(tag='full_validation_error', simple_value=validation_error_value.astype(np.float))
#          summary_writer.add_summary(vali_summ, step)
#          summary_writer.flush()
#        
#        else:
        _, validation_error_value, validation_loss_value = sess.run([self.val_op, self.vali_top1_error, self.vali_loss],\
                                                                    {self.vali_data_traj_placeholder: vali_batch_data_traj,\
                                                                     self.vali_labels_traj_placeholder: vali_batch_labels_traj,\
                                                                     self.vali_data_query_state_placeholder: vali_batch_data_query_state,\
                                                                     self.vali_labels_query_state_placeholder: vali_batch_labels_query_state,\
                                                                     self.lr_placeholder: self.INIT_LR})
        
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
      #     feed_dict = {self.train_data_traj_placeholder: train_batch_data,
      #                  self.train_labels_traj_placeholder: train_batch_labels,
      #                  self.vali_data_traj_placeholder: validation_batch_data,
      #                  self.vali_labels_traj_placeholder: validation_batch_labels,
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
      #         - param: global_step 
      #         - param: self.full_loss: 
      #             - The loss that includes both the loss and the regularized loss
      #             - comes from: self.full_loss = tf.add_n([loss] + regu_losses)
      #         - param: self.train_top1_error: 
      #             def _create_graphs(self):
      #                self.train_top1_error = self.top_k_error(predictions, self.train_labels_traj_placeholder, 1)
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
      # --- self.train_top1_error = self.top_k_error(predictions, self.train_labels_traj_placeholder, 1)
      # --- def top_k_error(self, predictions, labels, k):
      # - (2) The Top-1 error is the percentage of the time that the classifier 
      #       did not give the correct class the highest score.
      #
      # -----------------------------
      # feed_dict
      # -----------------------------
      # self.train_data_traj_placeholder: train_batch_data
      # - feed in the trajectories of the training batch
      # self.train_labels_traj_placeholder: train_batch_labels
      # - feed in the labels of the training batch
      # self.vali_data_traj_placeholder: validation_batch_data
      # - feed in the trajectories of the validation batch
      # self.vali_labels_traj_placeholder: validation_batch_labels
      # - feed in the labels of the validation batch
      # self.lr_placeholder: self.INIT_LR
      # - feed in the initial learning rate
      
      _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op, self.full_loss, self.train_top1_error],\
                                                           {self.train_data_traj_placeholder: train_batch_data_traj,\
                                                            self.train_labels_traj_placeholder: train_batch_labels_traj,\
                                                            self.train_data_query_state_placeholder: train_batch_data_query_state,\
                                                            self.train_labels_query_state_placeholder: train_batch_labels_query_state,\
                                                            self.lr_placeholder: self.INIT_LR})
      duration = time.time() - start_time

      if step % self.REPORT_FREQ == 0:
        summary_str = sess.run(summary_op,\
                               {self.train_data_traj_placeholder: train_batch_data_traj,\
                                self.train_labels_traj_placeholder: train_batch_labels_traj,\
                                self.train_data_query_state_placeholder: train_batch_data_query_state,\
                                self.train_labels_query_state_placeholder: train_batch_labels_query_state,\
                                self.vali_data_traj_placeholder: vali_batch_data_traj,\
                                self.vali_labels_traj_placeholder: vali_batch_labels_traj,\
                                self.vali_data_query_state_placeholder: vali_batch_data_query_state,\
                                self.vali_labels_query_state_placeholder: vali_batch_labels_query_state,\
                                self.lr_placeholder: self.INIT_LR})
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
  
      df_accuracy_all = self.evaluate_whole_data_set(files_traj = self.test_files_traj,\
                                                     data_traj = self.test_data_traj,\
                                                     labels_traj= self.test_labels_traj,\
                                                     data_query_state = self.test_data_query_state,\
                                                     labels_query_state = self.test_labels_query_state,\
                                                     batch_size = self.BATCH_SIZE_TEST,\
                                                     mode = 'test',
                                                     with_prednet = self.WITH_PREDNET)
      
      return df_accuracy_all
    
  def evaluate_on_validation_set(self):
      '''
      Evaluate a model with the validation data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.
      
      Returns:
        :df_accuracy_all: a dataframe with model accuracy.
      '''
      #TODO
      df_accuracy_all = self.evaluate_whole_data_set(files_traj = self.vali_files_traj,\
                                                     data_traj = self.vali_data_traj,\
                                                     labels_traj= self.vali_labels_traj,\
                                                     data_query_state = self.vali_data_query_state,\
                                                     labels_query_state = self.vali_labels_query_state,\
                                                     batch_size = self.BATCH_SIZE_VAL,\
                                                     mode = 'vali',
                                                     with_prednet = self.WITH_PREDNET)
      
      return df_accuracy_all
    
  def evaluate_whole_data_set(self, files_traj, data_traj, labels_traj, data_query_state, labels_query_state, batch_size, mode, with_prednet):
      '''
      Evaluate a model with a set of data (instead of a single batch).
      It will evaluate the data batch-by-batch and summarize the performance.
      It will return a dataframe with model accuracy.
      
      Args:
        :param files_traj: 
          the txt files_traj to be test (only used to compute the number of trajectories)
        :param data_traj: 
          the data_traj to be test the model on 
          (num_files, MAX_TRAJECTORY_SIZE, height, width, depth_trajectory)
        :param labels_traj: 
          If `with_prednet = False`, they are the ground truth labels to
          be test the model on (num_files, 1).
          If `with_prednet = True`, they are ignored.
        :param data_query_state: 
          If `with_prednet = True`, it is the query state
          of the new maze (num_files, height, width, depth_query_state).
          If `with_prednet = False`, they are ignored.
        :param labels_query_state:
          If `with_prednet = True`, they are the ground truth labels to
          be test the model on (num_files, 1).
          If `with_prednet = False`, they are ignored.
        :param batch_size: 
          the batch size
        :param mode: 
          should be either 'vali' or 'test'
        :param with_prednet:
          If `with_prednet = True`, then construct the complete model includeing
          both charnet and prednet.
          If `with_prednet = False`, then construct the partial model including
          only the charnet.        
      Returns:
        :df_accuracy_all:
          a dataframe with model accuracy.
      '''
      # pdb.set_trace()
     
      num_vali_files = len(files_traj)
      num_batches = num_vali_files // batch_size

      print('%i' %num_batches, mode, 'batches in total...')

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
          = self.batch_generator.generate_vali_batch(data_traj,\
                                     labels_traj,\
                                     data_query_state,\
                                     labels_query_state,\
                                     batch_size,\
                                     file_index = file_index)
  #        batch_data, batch_labels = self.batch_generator.generate_vali_batch(data, labels, batch_size, file_index)
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
          = self.batch_generator.generate_vali_batch(data_traj,\
                                     labels_traj,\
                                     data_query_state,\
                                     labels_query_state,\
                                     batch_size,\
                                     file_index = file_index)
  #        batch_data, batch_labels = self.batch_generator.generate_vali_batch(data, labels, batch_size, file_index)
          # pdb.set_trace()
          batch_prediction_array = sess.run(predictions,\
                                            feed_dict={data_traj_placeholder: batch_data_traj,\
                                                       data_query_state_placeholder: batch_data_query_state})
          # batch_prediction_array = (batch_size, num_classes)
          data_set_prediction_array = np.concatenate((data_set_prediction_array, batch_prediction_array))
          # vali_set_prediction_array will be size of (batch_size * num_batches, num_classes)
          data_set_ground_truth_labels = np.concatenate((data_set_ground_truth_labels, batch_labels_traj))
        # Model with both charnet and prednet
        
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
      #pdb.set_trace()
      return df_accuracy_all
    
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
    
