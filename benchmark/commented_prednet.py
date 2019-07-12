#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class PredNet(nnl.NeuralNetLayers): 

In this and subsequent experiments, 
we make three predictions: 
  (1)next-step action, 
  (2)which objects are consumed by the end of the episode, and
  (3) successor representations. 
  We use a shared torso for these predictions, from which separate heads branch off. 
  
  For the prediction torso, we 
    (1) spatialise echar,i, 
    (2) and concatenate with the query state; 
    (3) this is passed into a 5-layer resnet, with 32 channels, ReLU nonlinearities, and batch-norm.
    
  Consumption prediction head. 
  From the torso output: 
    （1) a 1-layer convnet with 32 channels and ReLUs, followed by average pooling, and 
     (2) a fully-connected layer to 4-dims, 
     (3) followed by a sigmoid. This gives the respective Bernoulli probabilities 
     that each of the four objects will be consumed by the end of the episode.
     [Unlike the paper, I replaced this sigmoid unit by a softmax unit.]
@author: Chuang, Yun-Shiuan
"""


import tensorflow as tf
#import numpy as np
#from tensorflow.contrib import rnn
import commented_nn_layers as nnl
# For debugging
import pdb
class PredNet(nnl.NeuralNetLayers): 
  
  def __init__(self):

      # hyperparameter for batch_normalization_layer()
      self.BN_EPSILON = 0.001
      
      # hyperparameter for create_variables()
      # this is especially for "tf.contrib.layers.l2_regularizer"
      self.WEIGHT_DECAY = 0.00002
      self.MAZE_DEPTH = 11
  
  def build_prednet(self,e_char, query_state_tensor, n, num_classes, reuse, train):
      '''
      Build the character net.
      
      :param e_char: the raw character embedding remained to be spatialized.
      :param query_state_tensor: the tensor for the query state
      :param n: the number of layers in the resnet
      :param num_classes: 
      :param reuse: ?
      :param train:  
      :return layers[-1]: "logits" is the output of the charnet (including ResNET and LSTM) and is the input for a softmax layer 
      '''
      # pdb.set_trace()
      layers = []
      
      # --------------------------------------------------------------    
      # Paper codes    
      # For the query_state_tensor:
      # Get the tensor size: (16, 12, 12, 11)
      # 16: batch size
      # 12: height
      # 12: width
      # 11: depth
      # --------------------------------------------------------------    
      # pdb.set_trace()
#      layers.append(query_state_tensor)
      # pdb.set_trace()
      batch_size, height, width, depth  = query_state_tensor.get_shape().as_list()
      
      # --------------------------------------------------------------    
      # Paper codes    
      # For the character embedding (e_char):
      # Get the tensor size: (16, 8)
      # 16: batch size
      # 8: the length of the character embedding
      # --------------------------------------------------------------    
      
      # pdb.set_trace()
      # pdb.set_trace()
      batch_size, embedding_length  = e_char.get_shape().as_list()
      
      # --------------------------------------------------------------    
      # Paper codes    
      # For e_char:
      # Spatialize e_char. (16, 8) -> (16, 12, 12, 8)
      # input: 
      # 16: batch size
      # 8: the length of the character embedding
      #
      # output:
      # 16: batch size
      # 12: height
      # 12: width
      # 8: the length of the character embedding
      # -------------------------------------------------------------- 
      e_char = tf.tile(e_char,[height, width,1])
      
      # --------------------------------------------------------------    
      # Paper codes    
      # Concatenate the query_state_tensor and the e_char    
      # (16, 12, 12, 8) + (16, 12, 12, 11) -> (16, 12, 12, 19)
      # -------------------------------------------------------------- 
      input_tensor = tf.stack(query_state_tensor, e_char)
      layers.append(input_tensor)
      # --------------------------------------------------------------
      
      # Paper codes    
      # (16, 12, 12, 19) -> (16, 12, 12, 32)
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
          
          # --------------------------------------------------------------
          # Paper codes
          # Do not need 'global average pooling'
          # already (16, 4)
          # --------------------------------------------------------------
  
          # ==============================================================
          # This section is to feed the result from LSTM to a FC layer
          # ==============================================================         
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