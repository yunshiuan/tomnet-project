import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

BN_EPSILON = 0.001
WEIGHT_DECAY = 0.00002


def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, is_fc_layer, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
    return new_variables

def batch_normalization_layer(input_layer, dimension):
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def conv_bn_relu_layer(input_layer, filter_shape, stride):
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape, is_fc_layer=False)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output

def residual_block(input_layer, output_channels):
    input_channel = input_layer.get_shape().as_list()[-1]
    stride = 1

    with tf.variable_scope('conv1_in_block'):
        conv1 = conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channels], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = conv_bn_relu_layer(conv1, [3, 3, output_channels, output_channels], stride)

    output = conv2 + input_layer
    
    return output

def average_pooling_layer(input_layer):
    pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return pooled_input

def lstm_layer(input_layer, mode, num_classes):
    num_hidden = 64
    #batch_size = 16
    out_channels = 11 #TODO: Change to depth of maze
    output_keep_prob = 0.8

    #Show the shape of the LSTM input layer
    #print(input_layer.get_shape().as_list())

    _, feature_h, feature_w, _ = input_layer.get_shape().as_list()
    
    lstm_input = tf.transpose(input_layer,[0,2,1,3])
    lstm_input = tf.reshape(lstm_input, [batch_size, feature_w, feature_h * out_channels])
    seq_len = tf.fill([lstm_input.get_shape().as_list()[0]], feature_w)
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    if mode == 'train':
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=output_keep_prob)

    #cell1 = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
    #if mode == 'train':
    #    cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=output_keep_prob)
    
    #stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, sequence_length=seq_len, initial_state=initial_state, dtype=tf.float32, time_major=False)
    outputs = tf.reshape(outputs, [-1, num_hidden])
    
    W = tf.get_variable(name='W_out', shape=[num_hidden, num_classes], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
    b = tf.get_variable(name='b_out', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer())

    #Linear output
    lstm_h = tf.matmul(outputs, W) + b
    shape = lstm_input.shape
    lstm_h = tf.reshape(lstm_h, [shape[0], -1, num_classes])
    return lstm_h

def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total?
    :return: output layer Y = WX + B
    '''
    
    input_dim = input_layer.get_shape().as_list()[-1]

    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], is_fc_layer=True, initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_layer, fc_w) + fc_b

    return fc_h

def build_charnet(input_tensor, n, num_classes, reuse, train):
    layers = []
       
    #Append the input tensor as first layer
    layers.append(input_tensor)
    
    #Add n residual layers
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            block = residual_block(layers[-1], 11) #TODO: Change here to the depth of the maze
            activation_summary(block)
            layers.append(block)
    
    #Add average pooling
    with tf.variable_scope('average_pooling', reuse=reuse):
        avg_pool = average_pooling_layer(block)
        layers.append(avg_pool)
    
    #Add LSTM layer
    with tf.variable_scope('LSTM', reuse=reuse):
        lstm = lstm_layer(layers[-1], train, num_classes)
        layers.append(lstm)
        
    #TODO:One convolutional layer


    #Fully connected
    with tf.variable_scope('fc', reuse=reuse):
        global_pool = tf.reduce_mean(layers[-1], [1])
        assert global_pool.get_shape().as_list()[-1:] == [num_classes]
        output = output_layer(global_pool, num_classes)
        layers.append(output)
    
    return layers[-1]

def build_pred_head(self):
    pass
