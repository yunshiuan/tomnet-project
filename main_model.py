import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import resnet as rn
import data_handler as dh
from tensorflow.contrib import rnn



class Model:
  HEIGHT = 12
  WIDTH = 12
  DEPTH = 11
  BATCH_SIZE_TRAIN = 32
  BATCH_SIZE_VAL = 32
  BATCH_SIZE_TEST = 32
  NUM_RESIDUAL_BLOCKS = 5
  TRAIN_EMA_DECAY = 0.95
  TRAIN_STEPS = 10000
  EPOCH_SIZE = 100 
  
  REPORT_FREQ = 100
  FULL_VALIDATION = False
  INIT_LR = 0.05

  DECAY_STEP_0 = 4000
  DECAY_STEP_1 = 8000
  
  NUM_CLASS = 4

  use_ckpt = False
  ckpt_path = 'cache_S002a_80000steps_2/logs/model.ckpt'
  train_path = 'cache_S002a_80000steps_2/train/'

  def __init__(self):
    #The data points must be given one by one here
    #But the whole trajectory must be given to the LSTM
    self.traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TRAIN, self.HEIGHT, self.WIDTH, self.DEPTH])
    self.goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_TRAIN])
    self.vali_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_VAL, self.HEIGHT, self.WIDTH, self.DEPTH])
    self.vali_goal_placeholder = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE_VAL])
    self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
            
  def _create_graphs(self):
    global_step = tf.Variable(0, trainable=False)
    validation_step = tf.Variable(0, trainable=False)
    
    logits = rn.build_charnet(self.traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=False, train=True)
    vali_logits = rn.build_charnet(self.vali_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=True)
    
    regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    #Training loss and error
    loss = self.loss(logits, self.goal_placeholder)
    self.full_loss = tf.add_n([loss] + regu_losses)
    predictions = tf.nn.softmax(logits)
    self.train_top1_error = self.top_k_error(predictions, self.goal_placeholder, 1)

    # Validation loss and error
    self.vali_loss = self.loss(vali_logits, self.vali_goal_placeholder)
    vali_predictions = tf.nn.softmax(vali_logits)
    self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_goal_placeholder, 1)

    # Define operations
    self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss, self.train_top1_error)
    self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)
    
    return
        
  def train(self):
    #Load data from tfrecord
    dir = os.getcwd() + '/S002a/'
    data_handler = dh.DataHandler(dir)

    train_data, vali_data, test_data, train_labels, vali_labels, test_labels = data_handler.parse_all_trajectories(dir)

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
      train_batch_data, train_batch_labels = self.generate_augment_train_batch(train_data, train_labels, self.BATCH_SIZE_TRAIN)
      validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data, vali_labels, self.BATCH_SIZE_VAL)

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
          checkpoint_path = os.path.join(self.train_path, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

          df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                          'validation_error': val_error_list})
          df.to_csv(self.train_path + '_error.csv')

    #model.test(test_data, test_labels)

  def test(self, test_trajectories, test_labels):
    '''
    This function is used to evaluate the test data. Please finish pre-precessing in advance
    :param test_image_array: 4D numpy array with shape [num_test_traj_steps, maze_height, maze_width, maze_depth]
    :return: the softmax probability with shape [num_test_traj_steps, num_labels]
    '''
    num_test_trajs = len(test_trajectories)
    num_batches = num_test_trajs // self.BATCH_SIZE_TEST
    remain_trajs = num_test_trajs % self.BATCH_SIZE_TEST
    print('%i test batches in total...' %num_batches)

    # Create the test image and labels placeholders
    self.test_traj_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.BATCH_SIZE_TEST, self.HEIGHT, self.WIDTH, self.DEPTH])

    # Build the test graph
    logits = rn.build_charnet(self.test_traj_placeholder, n=self.NUM_RESIDUAL_BLOCKS, num_classes=self.NUM_CLASS, reuse=True, train=False)
    predictions = tf.nn.softmax(logits)

    # Initialize a new session and restore a checkpoint
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()

    saver.restore(sess, os.path.join(self.train_path, 'model.ckpt-9999'))
    print('Model restored from ', os.path.join(self.train_path, 'model.ckpt-9999'))

    prediction_array = np.array([]).reshape(-1, self.NUM_CLASS)

    # Test by batches
    for step in range(num_batches):
      if step % 10 == 0:
          print('%i batches finished!' %step)
      offset = step * self.BATCH_SIZE_TEST
      test_traj_batch = test_trajectories[offset:offset+self.BATCH_SIZE_TEST, ...]

      batch_prediction_array = sess.run(predictions, feed_dict={self.test_traj_placeholder: test_traj_batch})
      prediction_array = np.concatenate((prediction_array, batch_prediction_array))

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
    
    matches = 0
    rounded_array = np.around(prediction_array,2).tolist()
    length = num_batches*self.BATCH_SIZE_TEST  
    for i in range(length):
      if(int(test_labels[i]+1) == rounded_array[i].index(max(rounded_array[i]))):
        matches += 1
    print('matches:', matches, '/', length)
    
    return prediction_array
  
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

    opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
    train_op = opt.minimize(total_loss, global_step=global_step)
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
    offset = np.random.choice(100 - vali_batch_size, 1)[0]
    vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
    vali_label_batch = vali_label[offset:offset+vali_batch_size]

    return vali_data_batch, vali_label_batch

  def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
    '''
    This function helps generate a batch of train data
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    
    offset = np.random.choice(self.EPOCH_SIZE - train_batch_size, 1)[0]
    batch_data = train_data[offset:offset + train_batch_size, ...]
    batch_label = train_labels[offset:offset + self.BATCH_SIZE_TRAIN]
    
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
    model = Model()
    model.train()
    
