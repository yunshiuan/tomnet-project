#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class CharNet(nnl.NeuralNetLayers): 

For the single trajectory τi in the past episode, the 
ToMnet forms the character embedding echar,i as follows. We
 (1) pre-process the data from each time-step by spatialising the actions,
 a(obs), concatenating these with the respective states, x(obs), 
 (2) passing through a 5-layer resnet, with 32 channels, ReLU nonlinearities,
 and batch-norm, followed by average pooling. 
 (3) We pass the results through an LSTM with 64 channels, 
 with a linear output to either a 2-dim or 8-dim echar,i (no substantial difference in results).
@author: Chuang, Yun-Shiuan
"""

import tensorflow as tf
#import numpy as np
#from tensorflow.contrib import rnn
import commented_nn_layers as nnl

# For debugging
import pdb
class CharNet(nnl.NeuralNetLayers): 
  
  def __init__(self):

      # hyperparameter for batch_normalization_layer()
      self.BN_EPSILON = 0.001
      
      # hyperparameter for create_variables()
      # this is especially for "tf.contrib.layers.l2_regularizer"
      self.WEIGHT_DECAY = 0.00002
      self.MAZE_DEPTH = 11
  
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
          # layers[-1] = step_wise_iput = (160, 12, 12, 11)
          #pdb.set_trace()
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