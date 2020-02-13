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
@author: Chuang, Yun-Shiuan; Edwinn
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
#      self.WEIGHT_DECAY = 0.00002
      self.WEIGHT_DECAY = 0.001
      self.MAZE_DEPTH = 11

  def build_charnet(self,input_tensor, n, num_classes, reuse, train):
      '''
      Build the character net.

      :param input_tensor:
      :param n: the number of layers in the resnet
      :param num_classes:
      :param reuse: ?
      :param train: If training, there will be dropout in the LSTM. For validation/testing,
        droupout won't be applied.
      :return layers[-1]: "logits" is the output of the charnet (including ResNET and LSTM) and is the input for a softmax layer
      '''
      # pdb.set_trace()
      layers = []

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
          #pdb.set_trace()
          conv_before_resnet = self.conv_layer_before_resnet(layers[-1])
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

      # --------------------------------------------------------------
      #Add average pooling
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
      with tf.variable_scope('average_pooling', reuse=reuse):
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

      # --------------------------------------------------------------
      #Fully connected layer
      # Paper codes
      # (16, 4) -> (16, 4)
      # def output_layer(self,input_layer, num_labels):
      #   '''
      #   A linear layer.
      #   :param input_layer: 2D tensor
      #   :param num_labels: int. How many output labels in total?
      #   :return: output layer Y = WX + B
      #   '''
      # --------------------------------------------------------------
      with tf.variable_scope('fc', reuse=reuse):
          # layers[-1] = (16, 64)
          output = self.output_layer(layers[-1], num_classes)
          # output = (16, 4)
          layers.append(output)

      return layers[-1]
