#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class NeuralNetLayers:

The parent class for both the character net and the predictio net.
@author: Chuang, Yun-Shiuan
"""

import tensorflow as tf
#import numpy as np
#from tensorflow.contrib import rnn

# For debugging
import pdb
class NeuralNetLayers:

  
  def __init__(self, BN_EPSILON, WEIGHT_DECAY, MAZE_DEPTH):
      #self.find_max_path(dir)
      # hyperparameter for batch_normalization_layer()
      self.BN_EPSILON = BN_EPSILON
      
      # hyperparameter for create_variables()
      # this is especially for "tf.contrib.layers.l2_regularizer"
      self.WEIGHT_DECAY = WEIGHT_DECAY
      self.MAZE_DEPTH = MAZE_DEPTH
    
      
  def activation_summary(self,x):
      tensor_name = x.op.name
      tf.summary.histogram(tensor_name + '/activations', x)
      tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  
  def create_variables(self,name, shape, is_fc_layer, initializer=tf.contrib.layers.xavier_initializer()):
      regularizer = tf.contrib.layers.l2_regularizer(scale=self.WEIGHT_DECAY)
      # pdb.set_trace()
      new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
      return new_variables
  
  def batch_normalization_layer(self, input_layer, dimension):
      mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
      beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
      bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, self.BN_EPSILON)
  
      return bn_layer
    
  def conv_layer(self, input_layer, filter_shape, stride):
      '''
      This layer is a pure convolutional layer (without bn, relu). 
      It is used for reshaping the tensor channels before entering the resnet.
      
      :param filter_shape: filter height, filter width, input_channel, output_channels
      '''    
      filter = self.create_variables(name='conv', shape=filter_shape, is_fc_layer=False)
      conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
  
      output = conv_layer
      return output
    
  def conv_layer_before_resnet(self, input_layer):
      '''
      Reshape the tensor channels before entering the resnet.
      '''
      stride = 1
      output_channels = 32
      #filter height, filter width, input_channel, output_channels
      filter_shape = [3, 3, input_layer.get_shape().as_list()[-1], output_channels]
      
      #output = conv_layer(input_layer, filter_shape, stride)
      
      filter = self.create_variables(name='conv', shape=filter_shape, is_fc_layer=False)
      conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
      output = conv_layer
     
      return output
      
  def conv_bn_relu_layer(self,input_layer, filter_shape, stride):
      '''
      This sub-block is the first sub-block in the residual connection of the residual block.
      
      :param filter_shape: filter height, filter width, input_channel, output_channels
      '''
      out_channel = filter_shape[-1] 
      
      # pdb.set_trace()
      
      filter = self.create_variables(name='conv', shape=filter_shape, is_fc_layer=False)
  
      # conv2d(input, filter, strides, padding)
      # input: 4D input tensor of shape [batch, in_height, in_width, in_channels]
      # filter: 4D filter tensor of shape [filter_height, filter_width, in_channels, out_channels]
      # - out_channels determin the number of channels
      # strides: Must have strides[0] = strides[3] = 1. 
      # - For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
      # padding: A string from: "SAME", "VALID".
      conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
      bn_layer = self.batch_normalization_layer(conv_layer, out_channel)
  
      output = tf.nn.relu(bn_layer)
      return output
    
  def conv_bn_no_relu_layer(self, input_layer, filter_shape, stride):
      '''
      This sub-block is the second sub-block in the residual connection of the residual block.
      This block is needed because ReLU should happen after addtion.
      
      :param filter_shape: filter height, filter width, input_channel, output_channels
      '''
      out_channel = filter_shape[-1]
      
      # pdb.set_trace()
      
      filter = self.create_variables(name='conv', shape=filter_shape, is_fc_layer=False)
  
      conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
      bn_layer = self.batch_normalization_layer(conv_layer, out_channel)
  
      output = bn_layer
      return output  
    
  def residual_block(self,input_layer, output_channels):
      '''
      Constructing a resudual block. Note that as in the paper, each residual block
      should have 32 channels (or 32 filers).
      
      :param input_layer: shape = (batch_size, height, width, depth)
      :param output_channels: 32 (as in the ToMNET paper)
      :return output: a[l+2] = g(z[l+2]+a[l]) = g(w[l+2]*a[l+1] + b[l+2] + a[l]) 
      '''
      input_channel = input_layer.get_shape().as_list()[-1]
      stride = 1
      # pdb.set_trace()
  
      # Note that conv_bn_relu_layer() includes both ReLU nonlinearities and batch-norm
      with tf.variable_scope('conv1_in_block'):
          # input_channel = 11
          # output_channels = 11
          conv1 = self.conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channels], stride)
  
      with tf.variable_scope('conv2_in_block'):
          # output_channels = 11
          # output_channels = 11
          conv2 = self.conv_bn_no_relu_layer(conv1, [3, 3, output_channels, output_channels], stride)
          
      # -------------------------------------------------
      # Make sure the second relu happens after addition
      # Following the original paper, should be “(a) original”: 
      # a[l+2] = g(z[l+2]+a[l]) = g(w[l+2]*a[l+1] + b[l+2] + a[l])
      # Reference: 
      # Identity Mappings in Deep Residual Networks (25 Jul 2016) 
      # (https://arxiv.org/pdf/1603.05027v3.pdf). See Figure 4.
      # -------------------------------------------------
      output = tf.nn.relu(conv2 + input_layer)
      
      return output
  
  def average_pooling_layer(self, input_layer):
      '''
      Collapse the spatical dimension.
      
      :param input layer: (steps, maze height, maze width, output channels). 
      The tensor will be averaged across the maze height and width dimensions
      and output a tensor with shape (steps, channels)
      '''
      # --------------------------------------------------------------
      # Edwinn's codes
      # Half the height and width of the tensor.
      # (16, 12, 12, 11) -> (16, 6, 6, 11)
      # --------------------------------------------------------------
      # pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
  
      # --------------------------------------------------------------
      # Paper codes
      # (160, 12, 12, 32) -> (160, 32)
      # --------------------------------------------------------------
      
      # tf.reduce_mean: Computes the mean of elements across dimensions of a tensor.
      # - param input_tensor: the output from the previous LSTM layer
      # - param axis: The dimensions to reduce
      global_pool = tf.reduce_mean(input_layer, [1,2])    
      return global_pool
  
  def lstm_layer(self, input_layer, train, num_classes):
      # --------------------------------------------------------------
      # Edwinn's codes
      # input_layer = (16, 6, 6, 11)
      # --------------------------------------------------------------
  #    num_hidden = 64 # Paper: 64
  #    batch_size = 16 # Paper: 16 
  #    out_channels = MAZE_DEPTH # note that output channels should not be the depth of the maze?
  #    output_keep_prob = 0.8 # This is for regularization during training
  #    
  #
  #    #Show the shape of the LSTM input layer
  #    #print(input_layer.get_shape().as_list())
  #    # input_layer.get_shape().as_list() = (16, 6, 6, 11)
  #    # 16: batch size (Tx for LSTM)
  #    # 6: featur_h after CNN
  #    # 6: feature_w after CNN
  #    # 11: number of channels after CNN (unaffected by CNN)
  #    feature_h, feature_w, _ = input_layer.get_shape().as_list() 
      # --------------------------------------------------------------
      # Paper
      # input_layer = (16, 10, 32)
      # --------------------------------------------------------------
  
      num_hidden = 64 # Paper: 64
      # batch_size = 16 # Paper: 16 # get from the input_layer directly
      # out_channels = MAZE_DEPTH # get from the input_layer directly
      output_keep_prob = 0.8 # This is for regularization during training
      
      # pdb.set_trace()
      ## Only for testing
      # input_layer = tf.zeros((16, 10, 6, 6, 11))
  
      #Show the shape of the LSTM input layer
      #print(input_layer.get_shape().as_list())
      
      # input_layer = (16, 10, 32)
      # 16: batch size (Tx for LSTM)
      # 10: time steps
      # 32: number of channels after CNN
      # pdb.set_trace()
      batch_size, time_steps, out_channels = input_layer.get_shape().as_list()
      
      # ==============================================================
      # This section is to reshape the tensor for LSTM 
      # ==============================================================
      
      # --------------------------------------------------------------
      # Edwinn's codes
      # (16, 6, 6, 11) -> (16, 6, 6x11)
      # --------------------------------------------------------------
  #    # Define the input for lstm
  #    # (1) input_layer.shape = (16, 6, 6, 11)
  #    lstm_input = tf.transpose(input_layer,[0,2,1,3]) # transpose the width and the height dimension
  #    # (2) lstm_input.shape = (16, 6, 6, 11)
  #    
  #    # Flatten the channel dimension so that 
  #    # the shape of each timestep of lstm_input changes from
  #    # 3-dim (w = 6, h = 6, c = 11) to
  #    # 2-dim (w = 6, h = 66)
  #    # (2) lstm_input.shape = (16, 6, 6, 11)
  #    lstm_input = tf.reshape(lstm_input, [batch_size, feature_w, feature_h * out_channels])
  #    # (3) lstm_input.shape = (16, 6, 66)
  #    # - 16: batch_size
  #    # - 6: feature_w
  #    # - 66: feature_h(6) * out_channels(11)
  
  
  
      # --------------------------------------------------------------
      # Paper:
      # No need to reshape becuase the spatial dimensions have
      # already been averaged out during the average pooling
      # (16, 10, 32) -> (16, 10, 32)
      #
      # (16, 10, 6, 6, 11):
      # 16: batch size
      # 10: time steps
      # 32: channels
      # 
      # (16, 10, 396):
      # 16: batch size
      # 10: time steps
      # 32: channels
      # --------------------------------------------------------------
      # Define the input for lstm
      lstm_input = input_layer
  
      # ==============================================================
      # This section is to perform LSTM 
      # ==============================================================
      # --------------------------------------------------------------
      # Edwinn's codes
      # (16, 6, 6x11) -> (16, 6, 64)
      # how: for each batch: W(64, 66) * x(66, 1) = a(64, 1)
      # --------------------------------------------------------------
  #    # seq_len.shape = (16)
  #    # An int32/int64 vector sized [batch_size]. 
  #    # Used to copy-through state and zero-out outputs when past a batch 
  #    # element's sequence length. So it's more for performance than correctness.
  #    seq_len = tf.fill([lstm_input.get_shape().as_list()[0]], feature_w)
  #    
  #    # cell:
  #    # the cell will be fed in to 
  #    # tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, sequence_length=seq_len, initial_state=initial_state, dtype=tf.float32, time_major=False)
  #    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
  #    
  #    if train:      
  #        # Using dropout for regularization during the RNN training
  #        # Dropout should not be used during validation and testing.
  #        #
  #        # rnn_cell.DropoutWrapper 
  #        # - param output_keep_prob: output keep probability (output dropout)
  #        #  if it is constant and 1, no output dropout will be added.
  #        # See: https://blog.csdn.net/abclhq2005/article/details/78683656
  #        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)
  #        
  #   
  #    #stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
  #
  #    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
  #    
  #    # (3) lstm_input.shape = (16, 6, 66)
  #    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, sequence_length=seq_len, initial_state=initial_state, dtype=tf.float32, time_major=False)
  #    
  #    # (4) outputs.shape = (16, 6, 64)
  #    # - cell:
  #    # - param lstm_input: shape = (16, 6, 66) 
  #    # current output: [16: batch_size, 6: width (after average pooling), 64: height (after average pooling) x channels]
  #    # should be: [16: batch_size, 16: max_time, ?: depth] (see the param 'time_major') #TODO
  #    # 
  #    # - param seq_len: shape = (16, )
  #    # - param initial_state:
  #    # - param time_major: False.
  #    # The shape format of the inputs and outputs Tensors. 
  #    # If true, these Tensors must be shaped [max_time, batch_size, depth].
  #    # If false, these Tensors must be shaped [batch_size, max_time, depth].
  #    # Using time_major = True is a bit more efficient because it avoids 
  #    # transposes at the beginning and end of the RNN calculation. 
  #    # However, most TensorFlow data is batch-major, 
  #    # so by default this function accepts input and emits output in batch-major form.
  #    # 
  #    # - return outputs: 
  #    # If time_major == False (default), this will be 
  #    # a Tensor shaped: [16: batch_size, 10: max_time, 64: cell.output_size].
  #    # current data: [16: batch_size, 6: width (after average pooling), 64: num_hidden]
  #    #   Note that the output size = num_hidden because the output of LSTM
  #    #   is 'a' (a = g(Waa * a + Wax * x + ba)) per se. Therefore, the 
  #    #   dimension of a is the dimension of the output.
  #    # should be: [16: batch_size, 10: max_time, 64: cell.output_size] #TODO
      
      # --------------------------------------------------------------
      # Paper:
      # (16, 10, 32) -> (16, 10, 64)
      #
      # (16, 10, 32):
      # 16: batch size
      # 10: time steps
      # 32: channels
      #
      # 16: batch size
      # 10: time steps
      # 64: hidden units
      # 1. for each x_i(t) (example_i's step_t):
      # a (64, 1) = W(64, 32) * x (32, 1) 
      # --------------------------------------------------------------
      # seq_len.shape = (16)
      # An int32/int64 vector sized [batch_size]. 
      # Used to copy-through state and zero-out outputs when past a batch 
      # element's sequence length. So it's more for performance than correctness.
      # (should be a vector of length batch_size, with each element represents the length of the
      # input)
      seq_len = tf.fill([batch_size], time_steps)
  
      # cell:
      # the cell will be fed in to 
      # tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, sequence_length=seq_len, initial_state=initial_state, dtype=tf.float32, time_major=False)
      cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
      
      if train:      
          # Using dropout for regularization during the RNN training
          # Dropout should not be used during validation and testing.
          #
          # rnn_cell.DropoutWrapper 
          # - param output_keep_prob: output keep probability (output dropout)
          #  if it is constant and 1, no output dropout will be added.
          # See: https://blog.csdn.net/abclhq2005/article/details/78683656
          cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)
          
     
      #stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
  
      initial_state = cell.zero_state(batch_size, dtype=tf.float32)
      
      # (3) lstm_input.shape = (16, 10, 32)
      # tf.nn.dynamic_rnn()
      # - param lstm_input = (batch_size, time_steps, input_filters) 
      # - param seq_len: shape = (16, )??Why did I set it to batch_size instead of num_time_steps? 
      # An int32/int64 vector sized [batch_size].  
      # Used to copy-through state and zero-out outputs when  
      # past a batch element's sequence length.  
      # So it's more for performance than correctness. 
      # initial_state = (batch_size, num_hiddens) 
      # - param time_major: False.
      # The shape format of the inputs and outputs Tensors. 
      # If true, these Tensors must be shaped [max_time, batch_size, depth].
      # If false, these Tensors must be shaped [batch_size, max_time, depth].
      # Using time_major = True is a bit more efficient because it avoids 
      # transposes at the beginning and end of the RNN calculation. 
      # However, most TensorFlow data is batch-major, 
      # so by default this function accepts input and emits output in batch-major form.
  
      outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, sequence_length=seq_len, initial_state=initial_state, dtype=tf.float32, time_major=False)
      
      # - return outputs: 
      # If time_major == False (default), this will be 
      # a Tensor shaped: [16: batch_size, 10: max_time, 64: cell.output_size].
      # Edwinn's codes: [16: batch_size, 6: width (after average pooling), 64: num_hidden]
      # Note that the output size = num_hidden because the output of LSTM
      # is 'a' (a = g(Waa * a + Wax * x + ba)) per se. Therefore, the 
      # dimension of a is the dimension of the output.
      # should be: [16: batch_size, 10: max_time, 64: cell.output_size]
      # - return final_state: 
      # outputs.shape = (batch_size, time_steps, num_hidden) 
      # final_state[-1] = (batch_size, num_hidden) 
      
      # (4) outputs.shape = (16, 10, 64)
      # Edwinn's codes: [16: batch_size, 6: width (after average pooling), 64: height (after average pooling) x channels]
      # should be: (batch_size, time_steps, input_channels) (see the param 'time_major')
      
  
      
      # ==============================================================
      # This section is to reshape for feeding in the linear layer
      # ==============================================================
      # --------------------------------------------------------------
      # Edwinn's codes
      # (16, 6, 64) -> (96, 64)
      # --------------------------------------------------------------
    
  #    # (4) outputs.shape = (16, 6, 64)
  #    outputs = tf.reshape(outputs, [-1, num_hidden])
  #    # (5) output.shape = (96, 64)
  
      # --------------------------------------------------------------
      # Paper:
      # (16, 10, 64) -> (16, 64)
      # 1. Take the output (batch = 16, num_hidden = 64) from the final block of the LSTM
      # --------------------------------------------------------------
      # (4) outputs.shape = (16, 10, 64)
      # outputs = outputs[:,-1,:]
      # (5) output.shape = (16, 64)
      
      # --------------------------------------------------------------
      # Paper: modified - v10
      # (16, 10, 64) -> (160, 64)
      # 1. Reshape to feed in a fc layer
      # --------------------------------------------------------------
      # (4) outputs.shape = (16, 10, 64)
      outputs = tf.reshape(outputs, [-1, num_hidden])
      # (5) output.shape = (160, 64)
  
      # ==============================================================
      # This section is to feed output from LSTM to a linear layer
      # ==============================================================
      # --------------------------------------------------------------
      # Edwinn's codes
      # (96, 64) -> (96, 4) -> (16, 6, 4)
      #
      # 1. For '(96, 64) -> (96, 4)':
      # 96 x 64 (num_hidden) -> 96 x 4 (num_classes)
      # by a linear layer (y = xW + b)
      # - x (96, 64)
      # - W (64, 4)
      # - b (4, )
      # - y (96, 4)
      # --------------------------------------------------------------
  
  #    #Linear output
  #    # W.shape = (64, 4)
  #    # b.shape = (4, )
  #    W = tf.get_variable(name='W_out', shape=[num_hidden, num_classes], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
  #    b = tf.get_variable(name='b_out', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer())
  
       # (6) output.shape = (96, 64)
  #    lstm_h = tf.matmul(outputs, W) + b
  #    # (7) lstm_h.shape = (96, 4)
  #
  #    shape = lstm_input.shape
  #    # lstm_input.shape = (16, 6, 66)
  #
  #    lstm_h = tf.reshape(lstm_h, [shape[0], -1, num_classes])
  #    # (8) lstm_h.shape = (16, 6, 4)
  #    return lstm_h
       
  #    # --------------------------------------------------------------
  #    # Paper:
  #    # (16, 64) -> (16, 4)
  #    # 1. For each batch, feed the output to a linear layer, W : (4, 64)
  #    # 2. the output from the linear layer should be (batch = 16, num_classes = 4)
  #    #
  #    # the linear layer (y = xW + b)
  #    # 1. For '(16, 64) -> (16, 4)':
  #    # 16 x 64 (num_hidden) -> 16 x 4 (num_classes)
  #    # by a linear layer (y = xW + b)
  #    # - x (16, 64)
  #    # - W (64, 4)
  #    # - b (4, )
  #    # - y (16, 4)
  #    # --------------------------------------
  #    
  #    # No need to reshape here
  #    ## (4) outputs.shape = (16, 64)
  #    # outputs = tf.reshape(outputs, [-1, num_hidden])
  #    ## (5) output.shape = (16, 4)
  #    
  #    # W.shape = (64, 4)
  #    # b.shape = (4, )
  #    W = tf.get_variable(name='W_out', shape=[num_hidden, num_classes], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
  #    b = tf.get_variable(name='b_out', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer())
  #    # pdb.set_trace()
  #
  #    #Linear output
  #    # (5) output.shape = (16, 64)
  #    linear_output = tf.matmul(outputs, W) + b
  #    # (6) linear_outputshape = (16, 4)
       
       
      # --------------------------------------------------------------
      # Paper: - modified v10
      # (160, 64) -> (160, 4) -> (16, 10, 4)
      # --------------------------------------#    
      # 1. Lump all steps together (160) and feed the output to a linear layer, W : (4, 64)
      # 2. the output from the linear layer should be (total_steps = 160, num_classes = 4)
      #
      # the linear layer (y = xW + b)
      # 1. For '(160, 64) -> (160, 4)':
      # 160 x 64 (num_hidden) -> 160 x 4 (num_classes)
      # by a linear layer (y = xW + b)
      # - x (160, 64)
      # - W (64, 4)
      # - b (4, )
      # - y (160, 4)
      # --------------------------------------#    
      
      # W = (64, 4)
      # b = (4, )
      W = tf.get_variable(name='W_out', shape=[num_hidden, num_classes], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
      b = tf.get_variable(name='b_out', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer())
      # pdb.set_trace()
  
      #Linear output
      # (5) output = (160, 64)
      linear_output = tf.matmul(outputs, W) + b
      # (6) linear_output = (160, 4)
      
  
      linear_output = tf.reshape(linear_output, [batch_size, time_steps, num_classes])
  #    # (8) lstm_h.shape = (16, 10, 4)
  #    return lstm_h
           
      # --------------------------------------
      # Paper: - modified v10
      # global average pooling:
      # average across the second axis:
      # --------------------------------------------------------------
      # (16, 10, 4) -> (16, 4) [6 2-d arrays reduce to 1 2-d array]
      # global_pool.shape = (16, 4)
      # --------------------------------------------------------------
      # tf.reduce_mean: Computes the mean of elements across dimensions of a tensor.
      # - param input_tensor: the output from the previous LSTM layer
      # - param axis: The dimensions to reduce
      # pdb.set_trace()
      linear_output = tf.reduce_mean(linear_output, [1])
      
      assert linear_output.get_shape().as_list()[-1:] == [num_classes]
       
      return linear_output
  
  def output_layer(self,input_layer, num_labels):
      '''
      A linear layer to resize the tensor.
      :param input_layer: 2D tensor
      :param num_labels: int. How many output labels in total?
      :return: output layer Y = WX + B
      '''
      
      input_dim = input_layer.get_shape().as_list()[-1]
  
      fc_w = self.create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      fc_b = self.create_variables(name='fc_bias', shape=[num_labels], is_fc_layer=True, initializer=tf.zeros_initializer())
      fc_h = tf.matmul(input_layer, fc_w) + fc_b
  
      return fc_h
  
  def build_charnet(self,input_tensor, n, num_classes, reuse, train):
      '''
      Build the character net.
      
      :param input_tensor:
      :param n: the number of layers in the resnet
      :param num_classes: 
      :param reuse: ?
      :param train:  
      :return layers[-1]: "logits" is the output of the charnet (including ResNET and LSTM) and is the input for a softmax layer 
      '''
      # pdb.set_trace()
      layers = []
      
      # --------------------------------------------------------------
      # Constants
      # -------------------------------------------------------------- 
      # resnet_input_channels = 32
         
      #Append the input tensor as the first layer
      # --------------------------------------------------------------
      # Edwinn's codes
      # --------------------------------------------------------------    
      # input_tensor.shape = (16, 12, 12, 11)
  #    layers.append(input_tensor)
      
      # --------------------------------------------------------------
      # Paper codes
      # Regard each step independently in the resnet
      # (16, 10, 12, 12, 11) -> (160, 12, 12, 11)
      # input_tensor.shape = (16, 10, 12, 12, 11)
      # 16: 16 trajectories
      # 10: each trajectory has 10 steps
      # 12, 12, 11: maze height, width, depth
      # --------------------------------------------------------------    
      # pdb.set_trace()
      layers.append(input_tensor)
      # pdb.set_trace()
      batch_size, trajectory_size, height, width, depth  = layers[-1].get_shape().as_list()
      # layers[-1] = input_tensor = (16, 10, 12, 12, 11)
      step_wise_input = tf.reshape(layers[-1], [batch_size * trajectory_size, height, width, depth])
      # resnet_iput = (160, 12, 12, 11)
      layers.append(step_wise_input)
  
      # --------------------------------------------------------------
      # Paper codes    
      # (160, 12, 12, 11) -> (160, 12, 12, 32)
      # Use 3x3 conv layer to shape the depth to 32
      # to enable resnet to work (addition between main path and residual connection)
      # --------------------------------------------------------------
      with tf.variable_scope('conv_before_resnet', reuse = reuse):
          # layers[-1] = step_wise_iput = (160, 12, 12, 32)
          conv_before_resnet = self.conv_layer_before_resnet(layers[-1])
          # conv_before_resnet = (160, 12, 12, 32)
          layers.append(conv_before_resnet)
          _, _, _, resnet_input_channels  = layers[-1].get_shape().as_list()
  
                        
      #Add n residual layers
      for i in range(n):
          with tf.variable_scope('conv_%d' %i, reuse=reuse):
  
              # --------------------------------------------------------------
              # Edwinn's codes
              # (16, 12, 12, 11) -> (16, 12, 12, 11)
              # layers[-1] = intput_tensor = (16, 12, 12, 11)
              # 16: 16 steps
              # 12, 12, 11: maze height, width, depth
              #
              # block = (16, 12, 12, 11)
              # 16: 16 steps
              # 12, 12, 11: maze height, width, output channels (SHOUlD be 32 as in the paper)
              # --------------------------------------------------------------
              
              # block = residual_block(layers[-1], MAZE_DEPTH) #resnet_output_channels) 
              # activation_summary(block)
              # layers.append(block)
              
              # --------------------------------------------------------------
              # Paper codes
              # (160, 12, 12, 32) -> (160, 12, 12, 32)
              # layers[-1] = intput_tensor = (16, 10, 12, 12, 32)
              # 160: 160 steps (16 trajectories x 10 steps/trajectory)
              # 10: each trajectory has 10 steps
              # 12, 12, 11: maze height, width, depth
              
              # block = (160, 12, 12, 32)
              # 160: 160 steps (16 trajectories x 10 steps/trajectory)
              # 12, 12, 32: maze height, width, output channels (as in the paper)
              # --------------------------------------------------------------
  
              #pdb.set_trace()
              # layers[-1] = (16, 10, 12, 12, 11)
              resnet_input = layers[-1]
              # resnet_input = (160, 12, 12, 11)
  
              block = self.residual_block(resnet_input, resnet_input_channels) 
              self.activation_summary(block)
              layers.append(block)
      
      #Add average pooling
      with tf.variable_scope('average_pooling', reuse=reuse):
          # --------------------------------------------------------------
          # Edwinn's codes
          # (16, 12, 12, 11) -> (16, 6, 6, 11)
          # layers[-1] = block = (16, 12, 12, 11) (after resnet)
          # 16: 16 steps
          # 12, 12, 11: feature height, width, channels (SHOUlD be 32 as in the paper)
          #
          # avg_pool = (16, 6, 6, 11)
          # 16: 16 steps
          # 6, 6, 11: feature height, width, output channels 
          # --------------------------------------------------------------
  
  #        avg_pool = average_pooling_layer(block)
  #        layers.append(avg_pool)
          
          # --------------------------------------------------------------
          # Paper codes
          # (160, 12, 12, 32) ->  (160, 32)
          # # collapse the spacial dimension
          #
          # layers[-1] = block = (160, 12, 12, 11)
          # 160: 160 steps (16 trajectories x 10 steps/trajectory)
          # 12, 12, 32: maze height, width, output channels (32 as in the paper)
          #  
          # avg_pool = (160, 32)
          # 160: 160 steps (16 trajectories x 10 steps/trajectory)
          # 32: output channels 
          # --------------------------------------------------------------
          avg_pool = self.average_pooling_layer(block)
          layers.append(avg_pool)
          
      
      
      #Add LSTM layer
      # pdb.set_trace()
  
      with tf.variable_scope('LSTM', reuse=reuse):
          # --------------------------------------------------------------
          # Edwinn's codes
          # (16, 6, 6, 11) -> (16, 6, 4)
          
          # layers[-1] = avg_pool = (16, 6, 6, 11)
          # 16: Tx
          # 6, 6, 11: the output width, height, and channels from average pooling
          
          # lstm = (16, 6, 4)
          # 16: Ty
          # 6: see lstm_layer(input_layer, train, num_classes) for details
          # 4: num_classes
          # --------------------------------------------------------------
          
  #        lstm = lstm_layer(layers[-1], train, num_classes)
  #        layers.append(lstm)
          
          # --------------------------------------------------------------
          # Paper codes
          # (160, 32) ->  (16, 4)  
          #
          # avg_pool = (160, 32)
          # 160: 160 steps (16 trajectories x 10 steps/trajectory)
          # 32: output channels 
          
          # lstm = (16, 4)
          # 16: batch_size (16 trajectories)
          # 4: num_classes     
          # --------------------------------------------------------------
  
          # --------------------------------------------------------------        
          # for testing only        
          # conclusion: reshaping twice result in the same np array
          # var_test = np.array(range(0,(5*10*6*6*11))).reshape(5, 10, 6, 6, 11)
          # var_test_reduce_dim = var_test.reshape(5 * 10, 6, 6, 11)
          # _, feature_h, feature_w, feature_d = var_test_reduce_dim.shape
          # var_test_resume_dim = var_test_reduce_dim.reshape(5, 10, feature_h, feature_w, feature_d)
          # np.array_equal(var_test_reduce_dim,var_test_resume_dim)
          # > False!??!!?? #TODO
          
          # layers[-1] = avg_pool = (160, 32)
          _, resnet_output_channels = layers[-1].get_shape().as_list()
          
          # layers[-1] = avg_pool = (160, 32)
          lstm_input = tf.reshape(layers[-1], [batch_size, trajectory_size, resnet_output_channels])
          # lstm_input = (16, 10, 32)
           
          # lstm_input = (16, 10, 32)
          lstm = self.lstm_layer(lstm_input, train, num_classes)
          # lstm = (16, 4)
          
          layers.append(lstm)        
  
      #Fully connected
      with tf.variable_scope('fc', reuse=reuse):
  
  
          # ==============================================================
          # This section is to change the tensor shape from (16, 6, 4) to (16, 4)
          # for FC later.
          # ==============================================================
          
          # global average pooling:
          # average across the second axis:
          # --------------------------------------------------------------
          # Edwinn's codes
          # from (16, 6, 4) to (16, 4) [6 2-d arrays reduce to 1 2-d array]
          # global_pool.shape = (16, 4)
          # --------------------------------------------------------------
  #        # tf.reduce_mean: Computes the mean of elements across dimensions of a tensor.
  #        # - param input_tensor: the output from the previous LSTM layer
  #        # - param axis: The dimensions to reduce
  #        global_pool = tf.reduce_mean(layers[-1], [1])
  #        assert global_pool.get_shape().as_list()[-1:] == [num_classes]
          
          # --------------------------------------------------------------
          # Paper codes
          # Do not need 'global average pooling'
          # already (16, 4)
          # --------------------------------------------------------------
  
          # ==============================================================
          # This section is to feed the result from LSTM to a FC layer
          # ==============================================================
          # --------------------------------------------------------------
          # Edwinn's codes
          # from (16, 6, 4) to (16, 4) [6 2-d arrays reduce to 1 2-d array]
          # global_pool.shape = (16, 4)
          # --------------------------------------------------------------        
          
  #        # def output_layer(input_layer, num_labels):
  #        # '''
  #        # :param input_layer: 2D tensor
  #        # :param num_labels: int. How many output labels in total?
  #        # :return: output layer Y = WX + B
  #        # '''
  #        
  #        # output.shape = (16, 4)
  #        output = output_layer(global_pool, num_classes)
  #        layers.append(output)
          
          # --------------------------------------------------------------
          # Paper codes
          # Do not need 'global average pooling'
          # already (16, 4)
          # --------------------------------------------------------------
          # def output_layer(input_layer, num_labels):
          # '''
          # :param input_layer: 2D tensor
          # :param num_labels: int. How many output labels in total?
          # :return: output layer Y = WX + B
          # '''
          
          # output.shape = (16, 4)
          output = self.output_layer(layers[-1], num_classes)
          layers.append(output)
      return layers[-1]