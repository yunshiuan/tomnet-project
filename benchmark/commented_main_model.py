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
import numpy as np
import pandas as pd
import tensorflow as tf
import commented_resnet as rn
import commented_data_handler as dh
import argparse
import itertools

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
  
  # tota number of minibatches used for training
  # (Paper: 2M minibatches, A.3.1. EXPERIMENT 1: SINGLE PAST MDP)
  TRAIN_STEPS = 200,000
  # the data size of an epoch (should equal to the traning set size)
  # e.g., given a full date set with 10,000 snapshots,
  # with a train:dev:test = 8:2:2 split,
  # EPOCH_SIZE should be 8,000 training files if there are 10,000 files
  EPOCH_SIZE = 800
  
  REPORT_FREQ = 100 # the frequency of writing the error to error.csv

  # TRUE: use the full data set for validation 
  # (but this would not be fair because a portion of the data has already been seen)
  # FALSE: data split using train:vali:test = 8:1:1
  FULL_VALIDATION = False 

  # Initial learning rate (LR) # paper: 10−4
  INIT_LR = 0.00001  # 10-5
  DECAY_STEP_0 = 10000 # LR decays for the first time (*0.9) at 10000th steps
  DECAY_STEP_1 = 15000 # LR decays for the second time (*0.9) at 15000th steps
  
  NUM_CLASS = 4 # number of unique classes in the training set

  use_ckpt = False
  ckpt_path = 'cache_S002a_10000files/logs/model.ckpt'
  train_path = 'cache_S002a_10000files/train/'

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
    dir = os.getcwd() + '/S002a/'
    # pdb.set_trace()
    data_handler = dh.DataHandler(dir)
    # For S002a:
    # Get the data by "data_handler.parse_trajectories(dir, mode=args.mode, shuf=args.shuffle)"
    # self.train_data.shape: (800, 12, 12, 11)
    # self.vali_data.shape: (100, 12, 12, 11)
    # self.test_data.shape: (100, 12, 12, 11)
    # self.train_labels.shape: (800, )
    # self.vali_labels.shape: (100, )
    # self.test_labels.shape: (100, )
    # len (files) = 100
    # Each data example is one trajectory (each contains 10 steps, MAX_TRAJECTORY_SIZE)
    
    # Note that all training examples are NOT shuffled randomly (by defualt)
    # during data_handler.parse_trajectories()
    
    self.train_data, self.vali_data, self.test_data, self.train_labels, self.vali_labels, self.test_labels, self.files = data_handler.parse_trajectories(dir, mode=args.mode, shuf=args.shuffle)

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

        num_examples_per_step = self.BATCH_SIZE_TRAIN
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

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
            
      if step == self.DECAY_STEP_0 or step == self.DECAY_STEP_1:
        self.INIT_LR = 0.1 * self.INIT_LR
        print('Learning rate decayed to ', self.INIT_LR)
        
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
    This function is used to evaluate the test data. Please finish pre-precessing in advance
    :param test_image_array: 4D numpy array with shape [num_test_traj_steps, maze_height, maze_width, maze_depth]
    :return: the softmax probability with shape [num_test_traj_steps, num_labels]
    '''
    # self.test_data = (100, 12, 12, 11) [when totol steps = 1,000 with 8:1:1 data split]
    num_test_trajs = len(self.test_data)/self.MAX_TRAJECTORY_SIZE
    # num_test_trajs = 10 [when totol steps = 1,000 with 8:1:1 data split]
  
    # num_batches = 10//batch_size = 0
    num_batches = int(num_test_trajs // self.BATCH_SIZE_TEST)
    # remain_trajs = num_test_trajs % self.BATCH_SIZE_TEST
    print('%i test batches in total...' %num_batches)
   
    # --------------------------------------------------------------
    # Paper
    # self.test_traj_placeholder  (for input_tensor)
    # - shape: 
    # (batch size = 16: batch_size, 10: MAX_TRAJECTORY_SIZE, HEIGHT = 12, WIDTH = 12, DEPTH = 11)
    # --------------------------------------------------------------
    
    # self.test_traj_placeholder.shape = (batch_size, 12, 12, 11)
    self.test_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TRAIN, self.MAX_TRAJECTORY_SIZE, self.HEIGHT, self.WIDTH, self.DEPTH])

    # Build the test graph
    if args.mode == 'all':
      # logits.shape = (batch_size, num_classes)
      logits = rn.build_charnet(self.test_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=False)
    else:
      logits = rn.build_charnet(self.test_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=False, train=False)

    # predictions.shape = (batch_size, num_classes)
    predictions = tf.nn.softmax(logits)

    # Initialize a new session and restore a checkpoint
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    saver.restore(sess, os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
    print('Model restored from ', os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))

    prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)

    # Test by batches
    #pdb.set_trace()
    for step in range(num_batches):
      if step % 10 == 0:
          print('%i batches finished!' %step)
      # pdb.set_trace()         
      # --------------------------------------------------------------
      # Edwinn's codes
      # generate test_traj_batch = (batch_size, height, width, depth)
      # --------------------------------------------------------------
      # offset = step * self.BATCH_SIZE_TEST
      # test_traj_batch = self.test_data[offset:offset+self.BATCH_SIZE_TEST, ...]

      # --------------------------------------------------------------
      # Paper
      # generate test_traj_batch = (batch_size * MAX_TRAJECTORY_SIZE, height, width, depth)
      # --------------------------------------------------------------
      

      # e.g., offset_batch_start_index = 2
      offset_batch_start_index = step
      # the ending batch
      # e.g., offset_batch_end_index = (2 + 16)  = 18
      # (note that this stopping index would be excluded by range())      
      offset_batch_end_index = (offset_batch_start_index + self.BATCH_SIZE_TEST)

      # e.g., offset_step_start_index = 2 * 10 = 20
      offset_step_start_index = offset_batch_start_index * self.MAX_TRAJECTORY_SIZE

      # e.g., offset_step_end_index = 18 * 10 = 180
      # (note that this stopping index would be excluded by range())
      offset_step_end_index = (offset_batch_end_index ) * self.MAX_TRAJECTORY_SIZE
      offset_step_range_index = range(offset_step_start_index, offset_step_end_index)

      test_traj_batch = self.test_data[offset_step_range_index, ...]

     
      # --------------------------------------------------------------
      # Paper
      # Reshape the batch data
      # (batch_size * MAX_TRAJECTORY_SIZE, height, width, depth) -> 
      # (batch_size, MAX_TRAJECTORY_SIZE, height, width, depth)
      # test_traj_batch = 
      # (batch_size * MAX_TRAJECTORY_SIZE, height, width, depth)
      #
      # test_traj_batch =
      # (batch_size, MAX_TRAJECTORY_SIZE, height, width, depth)
      # --------------------------------------------------------------
      # batch_data = (160, 6, 6, 11)

      test_traj_batch = test_traj_batch.reshape((self.BATCH_SIZE_TEST, self.MAX_TRAJECTORY_SIZE,
                                     self.HEIGHT, self.WIDTH, self.DEPTH))
      # vali_data_batch = (16, 10, 6, 6, 11)
      # pdb.set_trace()         

      # --------------------------------------------------------------
      # Paper
      # Making predictions
      # (batch_size, MAX_TRAJECTORY_SIZE, height, width, depth) ->
      # (batch_size, num_classes)
      #  test_traj_batch = 
      # (batch_size, MAX_TRAJECTORY_SIZE, height, width, depth)
      #
      # np.array(rounded_array).shape = 
      # (batch_size, num_classes)
      # --------------------------------------------------------------
      # predictions = (batch_size, num_classes)
      batch_prediction_array = sess.run(predictions, feed_dict={self.test_traj_placeholder: test_traj_batch})
      # batch_prediction_array = (batch_size, num_classes)

      # prediction_array = (0, num_classes)
      # concatenating it to collect results from diefference testing batches
      prediction_array = np.concatenate((prediction_array, batch_prediction_array))
      # after all interaion
      # prediction_array = (num_batches * batch_size, num_classes)

    # TODO: For now we dont have a way to handle batches of size != 32, so we are gonna have to skip the last few datapoints.
    '''
    if remain_trajs != 0:
      self.test_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_trajs, self.HEIGHT, self.WIDTH, self.DEPTH])
      # Build the test graph
      logits = rn.build_charnet(self.test_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=False)
      predictions = tf.nn.softmax(logits)

      test_traj_batch = test_trajectories[-remain_trajs:, ...]

      batch_prediction_array = sess.run(predictions, feed_dict={self.test_traj_placeholder: test_traj_batch})

      prediction_array = np.concatenate((prediction_array, batch_prediction_array))
    '''
    # prediction_array = (batch_size, num_classes)
    rounded_array = np.around(prediction_array,2).tolist()
    # rounded_array = (batch_size, num_classes)

    # length = (number of all testing trajectories) = (num_batches * batch_size)
    length = num_batches*self.BATCH_SIZE_TEST  
    
    # self.test_labels= (batch_size * MAX_TRAJECTORY_SIZE)
    # rounded_array = (batch_size, num_classes)
    # length = (number of all testing trajectories) = (num_batches * batch_size)
    # pdb.set_trace()
    self.match_estimation(self.test_labels, rounded_array, length) #print out performance metrics
    
    return prediction_array
  
  def match_estimation(self, labels, predictions, length):
    '''
    Evaluate model performance on the testing set
    
    :param labels: the ground truth labels (batch_size, num_classes)
    :param predicitons: the predicted labels with softmax probabilities (batch_size, num_classes)
    :param length: number of all testing trajectories (num_batches * batch_size)
    '''
    
    #Initialize zeroes for each possible arrangement
    # matches = 24 = 4! = num_classes!
    matches = [0 for item in range(math.factorial(self.NUM_CLASS))]
    
    for i in range(length): # number of all testing trajectories (num_batches * batch_size)
      #Initialize a 2d zeroes array
      # test = 24 x 4 = 4! x 4 = num_classes! x num_classes
      # filled by zeros
      test = [[0 for item in range(self.NUM_CLASS)] for item in range(math.factorial(self.NUM_CLASS))]
      # test = 24 x 4 = 4! x 4 = num_classes! x num_classes
      # all possible combination in 4!
      combinations = list(itertools.permutations(range(self.NUM_CLASS),self.NUM_CLASS))
      
      for j in range(math.factorial(self.NUM_CLASS)): # range(0, 24)
        for k in range(self.NUM_CLASS): # range(0, 4)
          test[j][k] = predictions[i][combinations[j][k]]
      
      for j in range(math.factorial(self.NUM_CLASS)): # range(0, 24)
        if int(labels[i]) == test[j].index(max(test[j])):
          matches[j] += 1

    best = matches.index(max(matches))
    print('Combination with best matches was ' + str(combinations[best]))
    print('Matches: ' + str(matches[best]) + '/' + str(length))
    print('Accuracy: ' + str(round(matches[best]*100/length,2)) + '%')
    
  
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
  
  def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
    '''
    If you want to use a random batch of validation data to validate instead of using the
    whole validation data, this function helps you generate that batch
    :param vali_data: 4D numpy array
    :param vali_label: 1D numpy array
    :param vali_batch_size: int
    :return: 4D numpy array and 1D numpy array
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
#    offset = np.random.choice(self.EPOCH_SIZE - vali_batch_size, 1)[0]
#    vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
#    vali_label_batch = vali_label[offset:offset+vali_batch_size]
    
    # --------------------------------------------------------------
    # Paper codes
    # Generate a batch. 
    # Each example is a trejectory.
    # Each batch contains 16 examples (trajectories). Each trajectory contains 10 steps.
    # batch_data shape = (16, 10, 6, 6, 11)
    # batch_label shape = (16, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    # the total number of batch equals the total number of steps devided by the steps fore each trajectory
    # (e.g., # validation steps = 1000, max_trajectory_size = 10, then total_number_batch = 100)
    total_number_vali_steps = vali_data.shape[0]
    total_number_batch = int(np.ceil(total_number_vali_steps/self.MAX_TRAJECTORY_SIZE))

    # Offsetting is to ensure that the batch ending index does not exceed the boundary of the epoch.
    # the starting batch #TODO: randomly select 16 batches instead of 16 continuous batches
    # e.g., offset_batch_start_index = 2
    offset_batch_start_index = np.random.choice(total_number_batch - vali_batch_size, 1)[0]
    # the ending batch
    # e.g., offset_batch_end_index = (2 + 16)  = 18
    # (note that this stopping index would be excluded by range())
    offset_batch_end_index = (offset_batch_start_index + vali_batch_size)

    # e.g., offset_step_start_index = 2 * 10 = 20
    offset_step_start_index = offset_batch_start_index * self.MAX_TRAJECTORY_SIZE
    # e.g., offset_step_end_index = 18 * 10 = 180
    # (note that this stopping index would be excluded by range())
    offset_step_end_index = (offset_batch_end_index ) * self.MAX_TRAJECTORY_SIZE
    offset_step_range_index = range(offset_step_start_index, offset_step_end_index)
    # pdb.set_trace()    
    
    # --------------------------------------------------------------
    # Select 16 random batches
    # (1000, 12, 12, 11) -> (160, 6, 6, 11)
    # --------------------------------------------------------------
    # vali_data = (1000, 12, 12, 11)
    batch_data = vali_data[offset_step_range_index , ...]
    # batch_data = (160, 6, 6, 11)
    
    # --------------------------------------------------------------
    # Reshape the batch data
    # (160, 6, 6, 11) -> (16, 10, 6, 6, 11)
    # --------------------------------------------------------------    
    # batch_data = (160, 6, 6, 11)
    vali_data_batch  = batch_data.reshape((vali_batch_size, self.MAX_TRAJECTORY_SIZE,
                                     self.HEIGHT, self.WIDTH, self.DEPTH))
    # vali_data_batch = (16, 10, 6, 6, 11)
    
    # --------------------------------------------------------------
    # Reshape the batch labels
    # (1000,) -> (160,)
    # --------------------------------------------------------------
    # vali_label = (1000,)    
    vali_label_batch  = vali_label[offset_step_range_index]
    # vali_label_batch = (160, ) 

    # --------------------------------------------------------------    
    # only retain 16 unique label (one for each batch)
    # (160,) -> (16,)
    # --------------------------------------------------------------    
    # vali_label_batch = (160,)
    vali_label_batch = vali_label_batch[0:-1:self.MAX_TRAJECTORY_SIZE]
    assert(vali_label_batch.shape[0]==vali_batch_size)
    # vali_label_batch = (16,)
    return vali_data_batch, vali_label_batch

  def generate_train_batch(self, train_data, train_labels, train_batch_size):
    '''
    This function helps generate a batch of train data
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    # -----------
    # numpy.random.choice:
    # If an ndarray, a random sample is generated from its elements. 
    # If an int, the random sample is generated as if a were np.arange(a)
    # -----------
    # > np.random.choice(self.EPOCH_SIZE - train_batch_size, 1)[0]
    # this generates an array with one random interger in it 
    # (from 0 to 84, 84: self.EPOCH_SIZE - train_batch_size)
    # -----------
    
    # --------------------------------------------------------------
    # Edwinn's codes
    # Generate a batch. 
    # Each batch is a step.
    # Each batch contains 16 steps (span from 2 trajectories, which
    # each contains 10 steps).
    # batch_data shape = (16, 6, 6, 11)
    # batch_label shape = (16, 1)
    # --------------------------------------------------------------
#    # Offsetting is to ensure that the batch ending index does not exceed the boundary of the epoch.
#    offset = np.random.choice(self.EPOCH_SIZE - train_batch_size, 1)[0]
#    # pdb.set_trace()
#    # subsetting a batch from traning data starting at index 'offet'
#    batch_data = train_data[offset:offset + train_batch_size, ...]
#    batch_label = train_labels[offset:offset + train_batch_size]
    
    # --------------------------------------------------------------
    # Paper codes
    # Generate a batch. 
    # Each example is a trejectory.
    # Each batch contains 16 examples (trajectories). Each trajectory contains 10 steps.
    # batch_data shape = (16, 10, 6, 6, 11)
    # batch_label shape = (16, 1)
    # --------------------------------------------------------------
    # pdb.set_trace()
    # the total number of batch equals the total number of steps devided by the steps fore each trajectory
    # (e.g., # training steps = 8000, max_trajectory_size = 10, then total_number_batch = 800)
    total_number_batch = int(np.ceil(self.EPOCH_SIZE/self.MAX_TRAJECTORY_SIZE)) 

    # Offsetting is to ensure that the batch ending index does not exceed the boundary of the epoch.
    # the starting batch
    # e.g., offset_batch_start_index = 2
    offset_batch_start_index = np.random.choice(total_number_batch - train_batch_size, 1)[0]
    # the ending batch
    # e.g., offset_batch_end_index = (2 + 16)  = 18
    # (note that this stopping index would be excluded by range())
    offset_batch_end_index = (offset_batch_start_index + train_batch_size)

    # e.g., offset_step_start_index = 2 * 10 = 20
    offset_step_start_index = offset_batch_start_index * self.MAX_TRAJECTORY_SIZE
    # e.g., offset_step_end_index = 18 * 10 = 180
    # (note that this stopping index would be excluded by range())
    offset_step_end_index = (offset_batch_end_index ) * self.MAX_TRAJECTORY_SIZE
    offset_step_range_index = range(offset_step_start_index, offset_step_end_index)
    
    # --------------------------------------------------------------
    # Select 16 random batches
    # (1000, 12, 12, 11) -> (160, 6, 6, 11)
    # --------------------------------------------------------------
    # vali_data = (1000, 12, 12, 11)
    batch_data = train_data[offset_step_range_index , ...]
    # batch_data = (160, 6, 6, 11)
    
    # --------------------------------------------------------------
    # Reshape the batch data
    # (160, 6, 6, 11) -> (16, 10, 6, 6, 11)
    # --------------------------------------------------------------
    # batch_data = (160, 6, 6, 11)
    batch_data = batch_data.reshape((train_batch_size, self.MAX_TRAJECTORY_SIZE,
                                     self.HEIGHT, self.WIDTH, self.DEPTH))
    # vali_data_batch = (16, 10, 6, 6, 11)

    
    # --------------------------------------------------------------
    # Reshape the batch labels
    # (1000,) -> (160,)
    # --------------------------------------------------------------
    # train_labels = (1000,)    
    batch_label = train_labels[offset_step_range_index]
    # batch_label = (160, ) 
    
    # --------------------------------------------------------------    
    # only retain 16 unique label (one for each batch)
    # (160,) -> (16,)
    # --------------------------------------------------------------   
    # batch_label = (160,)
    batch_label = batch_label[0:-1:self.MAX_TRAJECTORY_SIZE]
    # pdb.set_trace()
    assert(batch_label.shape[0]==train_batch_size)
    # batch_label = (16,)
    
    return batch_data, batch_label
    
  def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data, batch_label):
    '''
    Runs validation on all the validation datapoints
    :param loss: tensor with shape [1]
    :param top1_error: tensor with shape [1]
    :param session: the current tensorflow session
    :param vali_data: 4D numpy array
    :param vali_labels: 1D numpy array
    :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
    :param batch_label: 1D numpy array. training labels to feed the dict
    :return: float, float
    '''
    # This whole funciton is remained to be fixed ##TODO
    # (1) It should also comply with 
    # the batch = (16: batch_size, 10: MAX_TRAJECTORY_SIZE, HEIGHT = 12, WIDTH = 12, DEPTH = 11) rule
    num_batches = 10000 // self.BATCH_SIZE_VAL
    order = np.random.choice(10000, num_batches * self.BATCH_SIZE_VAL)
    vali_data_subset = vali_data[order, ...]
    vali_labels_subset = vali_labels[order]

    loss_list = []
    error_list = []

    for step in range(num_batches):
      offset = step * self.BATCH_SIZE_VAL
      feed_dict = {self.traj_placeholder: batch_data, self.goal_placeholder: batch_label,
        self.vali_traj_placeholder: vali_data_subset[offset:offset+self.BATCH_SIZE_VAL, ...],
        self.vali_goal_placeholder: vali_labels_subset[offset:offset+self.BATCH_SIZE_VAL],
        self.lr_placeholder: self.INIT_LR}
      loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
      loss_list.append(loss_value)
      error_list.append(top1_error_value)

    return np.mean(loss_list), np.mean(error_list)


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
    
