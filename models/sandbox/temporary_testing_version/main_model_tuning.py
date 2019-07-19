import os
import sys
import time
import math
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import resnet as rn
import data_handler as dh
import argparse
import itertools
import pdb
import re


 
class Model:
  HEIGHT = 12
  WIDTH = 12
  DEPTH = 11
  #BATCH_SIZE_TRAIN = 16
  #BATCH_SIZE_VAL = 16
  #BATCH_SIZE_TEST = 16
  NUM_RESIDUAL_BLOCKS = 5
  TRAIN_EMA_DECAY = 0.95
  #TRAIN_STEPS = 10000
  #EPOCH_SIZE = 100
  
  REPORT_FREQ = 100
  FULL_VALIDATION = False
  INIT_LR = 0.00001
  #DECAY_STEP_0 = 10000
  #DECAY_STEP_1 = 15000
  
  NUM_CLASS = 4

  use_ckpt = False
  #ckpt_path = 'cache_S002a_10000files/logs/model.ckpt'
  #train_path = 'cache_S002a_10000files/train/'

  def __init__(self, args, BATCH_SIZE_TRAIN,BATCH_SIZE_VAL, BATCH_SIZE_TEST, TRAIN_STEPS, EPOCH_SIZE, DECAY_STEP_0, DECAY_STEP_1, ckpt_fname, train_fname, sub_dir, subset_size):
    
    ckpt_path = ckpt_fname + '/logs/model.ckpt'
    train_path = train_fname + '/train/'
    
    self.ckpt_path = ckpt_path
    self.train_path = train_path

    self.BATCH_SIZE_TRAIN = BATCH_SIZE_TRAIN
    self.BATCH_SIZE_VAL = BATCH_SIZE_VAL
    self.BATCH_SIZE_TEST = BATCH_SIZE_TEST 
    self.TRAIN_STEPS = TRAIN_STEPS
    self.EPOCH_SIZE = EPOCH_SIZE
    self.DECAY_STEP_0 = DECAY_STEP_0
    self.DECAY_STEP_1 = DECAY_STEP_1
    
    rn.batch_size = BATCH_SIZE_TRAIN
    
    #The data points must be given one by one here?
    #But the whole trajectory must be given to the LSTM
    self.traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE_TRAIN, self.HEIGHT, self.WIDTH, self.DEPTH])
    self.goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE_TRAIN])
    self.vali_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE_VAL, self.HEIGHT, self.WIDTH, self.DEPTH])
    self.vali_goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE_VAL])
    self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    
    #Load data
    dir = os.path.join(os.getcwd(), sub_dir)
    data_handler = dh.DataHandler(dir)
    self.train_data, self.vali_data, self.test_data, self.train_labels, self.vali_labels, self.test_labels = data_handler.parse_trajectories(dir, mode=args.mode, shuf=args.shuffle, subset_size = subset_size)
    
            
  def _create_graphs(self):
    global_step = tf.Variable(0, trainable=False)
    validation_step = tf.Variable(0, trainable=False)
    
    logits = rn.build_charnet(self.traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=False, train=True)
    vali_logits = rn.build_charnet(self.vali_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=True)
    
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    #Training loss and error
    loss = self.loss(logits, self.goal_placeholder)
    self.full_loss = tf.add_n([loss] + regu_losses)
    predictions = tf.nn.softmax(logits, name = "op_predictions") #  to be restored
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
    #Build graphs
    self._create_graphs()
    
    # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
    # summarizing operations by running summary_op. Initialize a new session
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()
    sess = tf.Session()

    # If you want to load from a checkpoint
    if self.use_ckpt:
      saver.restore(sess, self.ckpt_path)
      print('Restored from checkpoint...')
    else:
      sess.run(init)
      
    # This summary writer object helps write summaries on tensorboard
    summary_writer = tf.summary.FileWriter(self.train_path, sess.graph)

    # These lists are used to save a csv file at last
    step_list = []
    train_error_list = []
    val_error_list = []
        
    print('Start training...')
    print('----------------------------')

    for step in range(self.TRAIN_STEPS):
      #Generate batches for training and validation
      #pdb.set_trace()
      train_batch_data, train_batch_labels = self.generate_train_batch(self.train_data, self.train_labels, self.BATCH_SIZE_TRAIN)
      validation_batch_data, validation_batch_labels = self.generate_vali_batch(self.vali_data, self.vali_labels, self.BATCH_SIZE_VAL)

      #Validate first?
      if step % self.REPORT_FREQ == 0:
        if self.FULL_VALIDATION:
          pass
          # validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss, top1_error=self.vali_top1_error, vali_data=vali_data, vali_labels=vali_labels, session=sess, batch_data=train_batch_data, batch_label=train_batch_labels)
    
          # vali_summ = tf.Summary()
          # vali_summ.value.add(tag='full_validation_error', simple_value=validation_error_value.astype(np.float))
          # summary_writer.add_summary(vali_summ, step)
          # summary_writer.flush()
        
        else:
          #pdb.set_trace()
          _, validation_error_value, validation_loss_value = sess.run([self.val_op, self.vali_top1_error, self.vali_loss], {self.traj_placeholder: train_batch_data, self.goal_placeholder: train_batch_labels, self.vali_traj_placeholder: validation_batch_data, self.vali_goal_placeholder: validation_batch_labels, self.lr_placeholder: self.INIT_LR})
        
          val_error_list.append(validation_error_value)
      
      start_time = time.time()
      

      #Actual training
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

        step_list.append(step)
        train_error_list.append(train_error_value)
            
      if step == self.DECAY_STEP_0 or step == self.DECAY_STEP_1:
        self.INIT_LR = 0.1 * self.INIT_LR
        print('Learning rate decayed to ', self.INIT_LR)

      # Save checkpoints every 10000 steps
      if step % 10000 == 0 or (step + 1) == self.TRAIN_STEPS:
          #pdb.set_trace()          
          checkpoint_path = os.path.join(self.train_path, 'model.ckpt')
          # Save the meta grach (to be restored)
          saver.save(sess, checkpoint_path, global_step=step)
          df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                  'validation_error': val_error_list})
          df.to_csv(self.train_path + '_error.csv')
          
  def test(self):
    '''
    This function is used to evaluate the validation and test data. Please finish pre-precessing in advance
    It will write a csv file with both validation and test perforance.
    '''

    num_test_trajs = len(self.test_data)
    num_batches = num_test_trajs // self.BATCH_SIZE_TEST
    # remain_trajs = num_test_trajs % self.BATCH_SIZE_TEST
    print('%i test batches in total...' %num_batches)

    # Create the test image and labels placeholders
    self.test_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TEST, self.HEIGHT, self.WIDTH, self.DEPTH])

    # Build the test graph
    if args.mode == 'all':
      logits = rn.build_charnet(self.test_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=False)
    else:
      logits = rn.build_charnet(self.test_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=False, train=False)
    predictions = tf.nn.softmax(logits)

    # Initialize a new session and restore a checkpoint
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    saver.restore(sess, os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))
    print('Model restored from ', os.path.join(self.train_path, 'model.ckpt-' + str(self.TRAIN_STEPS-1)))

    test_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)

    # Test by batches
    #pdb.set_trace()
    for step in range(num_batches):
      if step % 10 == 0:
          print('%i batches finished!' %step)
      offset = step * self.BATCH_SIZE_TEST
      test_traj_batch = self.test_data[offset:offset+self.BATCH_SIZE_TEST, ...]

      batch_prediction_array = sess.run(predictions, feed_dict={self.test_traj_placeholder: test_traj_batch})
      test_set_prediction_array = np.concatenate((test_set_prediction_array, batch_prediction_array))

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
    # --------------------------------------------------------------
    # My codes
    # Add a full validation set performance metric (instead of validation batch performance)
    # Call the full_validation() function
    # --------------------------------------------------------------
    # pdb.set_trace()
    df_vali_all = self.full_validation()
    
    # write the csv
    # df_test_v1.to_csv(self.train_path + '_test_accuracy_v2.csv')
    
    # --------------------------------------------------------------
    # Edwinn's codes
    # Test accuracy by match_estimation()
    # --------------------------------------------------------------
    test_set_rounded_array = np.around(test_set_prediction_array,2).tolist()
    length = num_batches*self.BATCH_SIZE_TEST  
    df_test_match_estimation = self.match_estimation(test_set_rounded_array, self.test_labels, length, 'test')

    # --------------------------------------------------------------
    # My codes
    # Test accuracy by definition
    # --------------------------------------------------------------
    df_test_proportion = self.proportion_accuracy(test_set_prediction_array, self.test_labels, length, 'test')

    # --------------------------------------------------------------
    # My codes
    # Combine all dfs into one
    # -------------------------------------------------------------- 
    #pdb.set_trace()

    df_test_all = df_test_proportion.append(df_test_match_estimation, ignore_index = True) 
    df_vali_and_test_all = df_vali_all.append(df_test_all)
    
    df_vali_and_test_all.to_csv(self.train_path + '_test_and_validation_accuracy.csv')

    return df_vali_and_test_all
  
  def full_validation(self):
      '''
      Runs validation on all the validation datapoints
      '''
      # pdb.set_trace()
     
      num_vali_steps = len(self.vali_data)
      num_batches = num_vali_steps // self.BATCH_SIZE_TEST
      # remain_trajs = num_vali_steps % self.BATCH_SIZE_TEST
      print('%i validation batches in total...' %num_batches)
  
      # Create the vali image and labels placeholders
      self.vali_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TEST, self.HEIGHT, self.WIDTH, self.DEPTH])
  
      # Build the vali graph
      logits = rn.build_charnet(self.vali_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=False)
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
      vali_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)
  
      # Test by batches
      #pdb.set_trace()
      for step in range(num_batches):
        if step % 10 == 0:
            print('%i batches finished!' %step)
        offset = step * self.BATCH_SIZE_TEST
        vali_traj_batch = self.vali_data[offset:offset+self.BATCH_SIZE_TEST, ...]
  
        batch_prediction_array = sess.run(predictions, feed_dict={self.vali_traj_placeholder: vali_traj_batch})
        # batch_prediction_array = (batch_size, num_classes)
        
        vali_set_prediction_array = np.concatenate((vali_set_prediction_array, batch_prediction_array))
        # vali_set_prediction_array will be size of (batch_size * num_batches, num_classes)
      
  
      # --------------------------------------------------------------
      # Edwinn's codes
      # Test accuracy by match_estimation()
      # --------------------------------------------------------------
      # vali_set_prediction_array = (batch_size * num_batches) x num_classes
      # length = (batch_size * num_batches)
      rounded_array = np.around(vali_set_prediction_array,2).tolist()
      length = num_batches*self.BATCH_SIZE_TEST  
      df_vali_match_estimation = self.match_estimation(rounded_array, self.vali_labels, length, 'vali')
      
      # --------------------------------------------------------------
      # My codes
      # Test accuracy by definition
      # --------------------------------------------------------------
      # pdb.set_trace()
      df_vali_proportion = self.proportion_accuracy(vali_set_prediction_array, self.vali_labels, length, 'vali')
  
      # --------------------------------------------------------------
      # My codes
      # Combine all dfs into one
      # -------------------------------------------------------------- 
      #pdb.set_trace()
  
      df_vali_all = df_vali_proportion.append(df_vali_match_estimation, ignore_index = True) 
  
      return df_vali_all

  def match_estimation(self,predictions, labels, length, mode):
    '''
    Evaluate model accuracy defined by Edwinn's method.
    Return a df that contains the accuracy metric.
    
    :param labels: ground truth labels (including both in-batch and out-of-batch
    labels. Note that only in-batch labels (size = length) are tested because 
    they have corresponding predicted labels.
    :param predicitons: predicted labels (num_batches * batch_size, num_classes).
    :param length: (num_batches * batch_size, 1). This defines the number of 
    labels that are in batches. Note that some remaining labels are not
    included in batches.
    :param mode: should be either 'vali' or 'test'
    :return df_summary: a a dataframe that stores the acuuracy metrics
    '''
    
    #Initialize zeroes for each possible arrangement
    matches = [0 for item in range(math.factorial(self.NUM_CLASS))]
    
    for i in range(length):
      #Initialize a 2d zeroes array
      test = [[0 for item in range(self.NUM_CLASS)] for item in range(math.factorial(self.NUM_CLASS))]
      combinations = list(itertools.permutations(range(self.NUM_CLASS),self.NUM_CLASS))
      for j in range(math.factorial(self.NUM_CLASS)):
        for k in range(self.NUM_CLASS):
          test[j][k] = predictions[i][combinations[j][k]]
      
      for j in range(math.factorial(self.NUM_CLASS)):
        if int(labels[i]) == test[j].index(max(test[j])):
          matches[j] += 1

    best = matches.index(max(matches))
    print('\n' + str(mode) +': match_estimation()')
    print( 'Combination with best matches was ' + str(combinations[best]))
    print('Matches: ' + str(matches[best]) + '/' + str(length))
    print('Accuracy: ' + str(round(matches[best]*100/length,2)) + '%')
    df_summary = pd.DataFrame(data={'matches':str(str(matches[best]) + '/' + str(length)),
                                    'accurary':str(str(round(matches[best]*100/length,2)) + '%'),
                                    'mode': str(mode) + '_match_estimation'},
                      index = [0])
    ## write the csv
    #df.to_csv(self.train_path + '_test_accuracy_v1.csv')
    return df_summary
  
  def proportion_accuracy(self, prediction_array, labels, length, mode):
    '''
    Evaluate model accuracy defined by proportion (num_matches/num_total).
    Return a df that contains the accuracy metric.
    
    :param prediction_array: a tensor with (num_batches * batch_size, num_classes).
    :param labels: ground truth labels (including both in-batch and out-of-batch
    labels. Note that only in-batch labels (size = length) are tested because 
    they have corresponding predicted labels.
    :param mode: should be either 'vali' or 'test'
    :param length: (num_batches * batch_size, 1). This defines the number of 
    labels that are in batches. Note that some remaining labels are not
    included in batches.
    :return df_summary: a a dataframe that stores the acuuracy metrics
    '''
    total_predictions = len(prediction_array)
    # match_predictions
    predicted_labels = np.argmax(prediction_array,1)
    
    # Retrieve corresponding labels
    groud_truth_labels = labels.astype(int)[0:length]
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
    batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
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
    offset = 0 #np.random.choice(100 - vali_batch_size, 1)[0]
    vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
    vali_label_batch = vali_label[offset:offset+vali_batch_size]

    return vali_data_batch, vali_label_batch

  def generate_train_batch(self, train_data, train_labels, train_batch_size):
    '''
    This function helps generate a batch of train data

    Args:
      :param train_data: 4D numpy array
      :param train_labels: 1D numpy array
      :param train_batch_size: int
      
    Return: 
      :batch_data: augmented train batch data. 4D numpy array.
      :batch_label: augmented train batch labels. 1D numpy array      
    '''
    offset = np.random.choice(self.EPOCH_SIZE - train_batch_size, 1)[0]
    batch_data = train_data[offset:offset + train_batch_size, ...]
    batch_label = train_labels[offset:offset + self.BATCH_SIZE_TRAIN]
    
    return batch_data, batch_label
  
  def predict_preference_ranking(self, ckpt_meta_file, dir_testing_maze_txt):
    '''
    This function predicts the target preference.
    See this tutorial for detail about restoring model:
    https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
  
    Args:
      :param ckpt_meta_file: The model (full path) meta file to be evaluated.
      :testing_maze_txt: The dir contains txt files (full path) that contain the maze where 
      agent is in the center and target all of the same distancs to the agent.
      
    Return: 
      :predicted_preference: softmax probability of (num_classes, 1).
    '''
    # ------------------------------------------------
    # Process the data for making preference ranking predictions
    # ------------------------------------------------
    # pdb.set_trace()
    directory = os.path.join(os.getcwd(), dir_testing_maze_txt)
    data_handler = dh.DataHandler(directory)
    
    # Only read the txt files
    files = os.listdir(directory)
    # Filter out the csv file (only read the txt files) 
    r = re.compile(".*.txt") 
    files = list(filter(r.match, files)) # Read Note  
    #file_full_path = os.path.join(directory,files[0])
    self.prediction_data = data_handler.parse_subset_maze_with_no_steps(directory, files)
    pdb.set_trace()
   
    # ------------------------------------------------
    # Import the saved graph and restore the parameters
    # ------------------------------------------------
    # pdb.set_trace()

    # Create the test image and labels placeholders
#    self.test_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TEST, self.HEIGHT, self.WIDTH, self.DEPTH])
#    sess = tf.Session()   
#    saver = tf.train.import_meta_graph(ckpt_meta_file)
#    # get rid of '.meta'
#    path_formated_for_restore = re.sub(".meta$","",ckpt_meta_file)
#    path_formated_for_restore = os.path.join('.',path_formated_for_restore)
#    saver.restore(sess,(path_formated_for_restore))
#      
#    graph = tf.get_default_graph()
#    logits = graph.get_tensor_by_name("fc")
#    predictions = tf.nn.softmax(logits)
    ## Build graphs
    #self._create_graphs()
    

    # Initialize a new session and restore a checkpoint
    # initialize
    
    sess = tf.Session()
#    init = tf.initialize_all_variables()
#    sess.run(init)
    
    # Load in the meta graph and restore weights
    saver = tf.train.import_meta_graph(ckpt_meta_file)    
    path_formated_for_restore = re.sub(".meta$","",ckpt_meta_file)
    path_formated_for_restore = os.path.join('.',path_formated_for_restore)
    saver.restore(sess,(path_formated_for_restore))
    
    



    # Now, let's access and create placeholders variables and
    graph = tf.get_default_graph()
    self.test_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TEST, self.HEIGHT, self.WIDTH, self.DEPTH])

    # create feed-dict to feed new data
    op_predictions = graph.get_tensor_by_name("op_predictions:0")

    #Now, access the op that you want to run. 
    
    # Initialize a new session and restore a checkpoint
    print('\n ----------------------------------------')
    print('Model restored from: \n', ckpt_meta_file)
    print('\n ----------------------------------------')
    print('Prediction preference based on: \n  ', dir_testing_maze_txt)
    print('\n ----------------------------------------')

    # ------------------------------------------------
    # Make predictions
    # ------------------------------------------------
    # pdb.set_trace()
    num_test_trajs = len(self.prediction_data)
    num_batches = num_test_trajs // self.BATCH_SIZE_TEST
    # remain_trajs = num_test_trajs % self.BATCH_SIZE_TEST
    print('%i prediction batches in total...' %num_batches)
    
    test_set_prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)
  
    # Test by batches
    pdb.set_trace()
    for step in range(num_batches):
      if step % 10 == 0:
          print('%i batches finished!' %step)
          

      offset = step * self.BATCH_SIZE_TEST
      test_traj_batch = self.prediction_data[offset:offset+BATCH_SIZE_TEST, ...]
  
      batch_prediction_array = sess.run(op_predictions, feed_dict={self.test_traj_placeholder: test_traj_batch})
      test_set_prediction_array = np.concatenate((test_set_prediction_array, batch_prediction_array))
  
    return

    
if __name__ == "__main__":
    tf.reset_default_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', help='all: train and test, train: only train, test: only test')
    parser.add_argument('--shuffle', type=str, default=False, help='shuffle the data for more random result')
    
    args = parser.parse_args()
    
    for times in range(1, 2, 1):
        BATCH_SIZE_TRAIN = 96
        BATCH_SIZE_VAL = BATCH_SIZE_TRAIN
        BATCH_SIZE_TEST = BATCH_SIZE_TRAIN
        TRAIN_STEPS = 50000
        EPOCH_SIZE = 78600
        DECAY_STEP_0 = 10000
        DECAY_STEP_1 = 15000
        ckpt_fname = 'training_result/caches/cache_S002a_v14_commit_1f58ab_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5_' + str(times)
        train_fname = 'training_result/caches/cache_S002a_v14_commit_1f58ab_epoch78600_tuning_batch96_train_step_2M_INIT_LR_10-5_' + str(times)
        sub_dir='/../S002a/'
#        sub_dir='/../../S002a_1000files/'

        subset_size = -1
        # path_mode = '.'
        path_mode =  '../test_on_human_data/' # Necessary when the output dir and script dir is different
        ckpt_fname = 'training_result/caches/cache_S030_v7_commit_0050d9_epoch80000_tuning_batch96_train_step_0.5M_INIT_LR_10-5_' + str(times)
        train_fname = 'training_result/caches/cache_S030_v7_commit_0050d9_epoch80000_tuning_batch96_train_step_0.5M_INIT_LR_10-5_' + str(times)
        sub_dir='data/processed/S030/'
        ckpt_fname = os.path.join(path_mode,ckpt_fname)
        train_fname = os.path.join(path_mode,train_fname)
        sub_dir = os.path.join(path_mode,sub_dir)
        # pdb.set_trace()
        


        model = Model(args,BATCH_SIZE_TRAIN,BATCH_SIZE_VAL, BATCH_SIZE_TEST, TRAIN_STEPS, EPOCH_SIZE,DECAY_STEP_0, DECAY_STEP_1, ckpt_fname, train_fname, sub_dir, subset_size)
        # pdb.set_trace()

        #############
        # Make prediction based on a trained model
        ckpt_meta_file = 'training_result/caches/cache_S002a_v7_commit_6f14c6_epoch80000_tuning_batch96_train_step_2M_INIT_LR_10-5_1/train/model.ckpt-199999.meta'
        dir_testing_maze_txt = 'data_for_making_preference_predictions'
        model.predict_preference_ranking(ckpt_meta_file, dir_testing_maze_txt)
        #############
        if args.mode == 'train' or args.mode == 'all':
            model.train()
        if args.mode == 'test' or args.mode == 'all':
            model.test()
            
        #reset variables
        tf.reset_default_graph()
        #del conv
        #del fc_weights
        #del fc_bias


