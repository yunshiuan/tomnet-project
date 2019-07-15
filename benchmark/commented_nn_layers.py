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
      No ReLU and BN. Only a 3x3 conv filter.
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
    
  def conv_prediction_head_layer(self, input_layer, filter_shape, stride):
      '''
      This conv layer is for the prediction head of the prednet, 
      including only conv and ReLU.
      
      :param filter_shape: filter height, filter width, input_channel, output_channels
      '''
      stride = 1
      output_channels = 32
      #filter height, filter width, input_channel, output_channels
      filter_shape = [3, 3, input_layer.get_shape().as_list()[-1], output_channels]
      
      #output = conv_layer(input_layer, filter_shape, stride)
      
      filter = self.create_variables(name='conv', shape=filter_shape, is_fc_layer=False)
      conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
      output = tf.nn.relu(conv_layer)     
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
      This block is needed because ReLU should happen after addition.
      
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
            
      # Define the input for lstm
      lstm_input = input_layer
  
      # ==============================================================
      # This section is to perform LSTM 
      # ==============================================================    
      
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
      # Paper: modified - v13
      # (16, 4) 
      # 1. No need to resize. It is already with the correct size.
      # --------------------------------------------------------------
      final_state = final_state[1]  
      
      # ==============================================================
      # This section is to feed output from LSTM to a linear layer
      # ==============================================================      
      # --------------------------------------------------------------
      # Paper: modified - v13
      # (16, 64)
      # 1. Directly output the final state
      # --------------------------------------------------------------
      # (4) final_state.shape = (16, 64)
      linear_output = final_state
      # (5) linear_output.shape = (16, 64)
      #pdb.set_trace()
      assert linear_output.get_shape().as_list() == [batch_size, num_hidden]            
      return linear_output
  
  def output_layer(self,input_layer, num_labels):
      '''
      A linear layer/ fully connected layer/ dense layer.
      :param input_layer: 2D tensor
      :param num_labels: int. How many output labels in total?
      :return: output layer Y = WX + B
      '''
      
      input_dim = input_layer.get_shape().as_list()[-1]
  
      fc_w = self.create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      fc_b = self.create_variables(name='fc_bias', shape=[num_labels], is_fc_layer=True, initializer=tf.zeros_initializer())
      fc_h = tf.matmul(input_layer, fc_w) + fc_b
  
      return fc_h
