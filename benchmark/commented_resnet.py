import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# For debugging

import pdb

# hyperparameter for batch_normalization_layer()
BN_EPSILON = 0.001

# hyperparameter for create_variables()
# this is especially for "tf.contrib.layers.l2_regularizer"
WEIGHT_DECAY = 0.00002
MAZE_DEPTH = 11

def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, is_fc_layer, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
    # pdb.set_trace()
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
    
    # pdb.set_trace()
    
    filter = create_variables(name='conv', shape=filter_shape, is_fc_layer=False)

    # conv2d(input, filter, strides, padding)
    # input: 4D input tensor of shape [batch, in_height, in_width, in_channels]
    # filter: 4D filter tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # - out_channels determin the number of channels
    # strides: Must have strides[0] = strides[3] = 1. 
    # - For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1].
    # padding: A string from: "SAME", "VALID".
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output
def conv_bn_no_relu_layer(input_layer, filter_shape, stride):
    out_channel = filter_shape[-1]
    
    # pdb.set_trace()
    
    filter = create_variables(name='conv', shape=filter_shape, is_fc_layer=False)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = bn_layer
    return output  
def residual_block(input_layer, output_channels):
    '''
    Constructing a resudual block. Note that as in the paper, each residual block
    should have 32 channels (or 32 filers), but currently we use 11 channels to 
    correspond to the number of channels of the input tensor (to ensure addtion of
    "main path" and "residual connection")
    
    :param input_layer: shape = (16, 12, 12, 11)
    :param output_channels: 11 (note: different from the ToMNET paper)
    :return output: a[l+2] = g(z[l+2]+a[l]) = g(w[l+2]*a[l+1] + b[l+2] + a[l]) 
    '''
    input_channel = input_layer.get_shape().as_list()[-1]
    stride = 1
    # pdb.set_trace()

    # Note that conv_bn_relu_layer() includes both ReLU nonlinearities and batch-norm
    with tf.variable_scope('conv1_in_block'):
        # input_channel = 11
        # output_channels = 11
        conv1 = conv_bn_relu_layer(input_layer, [3, 3, input_channel, output_channels], stride)

    with tf.variable_scope('conv2_in_block'):
        # output_channels = 11
        # output_channels = 11
        conv2 = conv_bn_no_relu_layer(conv1, [3, 3, output_channels, output_channels], stride)
        
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

def average_pooling_layer(input_layer):
    pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return pooled_input

def lstm_layer(input_layer, train, num_classes):
    # --------------------------------------------------------------
    # Edwinn's codes
    # --------------------------------------------------------------
#    num_hidden = 64 # Paper: 64
#    batch_size = 16 # Paper: 16 
#    out_channels = MAZE_DEPTH 
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
    # --------------------------------------------------------------

    num_hidden = 64 # Paper: 64
    # batch_size = 16 # Paper: 16 # get from the input_layer directly
    # out_channels = MAZE_DEPTH # get from the input_layer directly
    output_keep_prob = 0.8 # This is for regularization during training
    
    pdb.set_trace()
    # Only for testing
    input_layer = tf.zeros((16, 10, 6, 6, 11))

    #Show the shape of the LSTM input layer
    #print(input_layer.get_shape().as_list())
    # input_layer.get_shape().as_list() = (16, 6, 6, 11)
    # 16: batch size (Tx for LSTM)
    # 6: featur_h after CNN
    # 6: feature_w after CNN
    # 11: number of channels after CNN (unaffected by CNN)
    batch_size, time_steps, feature_h, feature_w, maze_depth = input_layer.get_shape().as_list()
    out_channels = maze_depth # note that output channels should not be the depth of the maze? #TODO
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
    # (16, 10, 6, 6, 11) -> (16, 10, 396)
    #
    # (16, 10, 6, 6, 11):
    # 16: batch size
    # 10: time steps
    # 6, 6, 11: maze tensor after average pooling
    # 
    # (16, 10, 396):
    # 16: batch size
    # 10: time steps
    # 396 = 6 x 6 x 11
    # --------------------------------------------------------------
    # Define the input for lstm
    # (1) input_layer.shape = (16, 10, 6, 6, 11)
    pdb.set_trace()
    # No need to transpose width and height?
    # lstm_input = tf.transpose(input_layer,[0,1,3,2,4])  # transpose the width and the height dimension
    lstm_input = input_layer
    # (2) lstm_input.shape = (16, 10, 6, 6, 11)
    
    # Flatten the channel dimension so that 
    # the shape of each x_i 's time step t of lstm_input changes from
    # 3-dim (w = 6, h = 6, c = 11) to
    # 1-dim (depth = 6 x 6 x 11)
    # (2) lstm_input.shape = (16, 10, 6, 6, 11)
    lstm_input = tf.reshape(lstm_input, [batch_size, time_steps, feature_w * feature_h * out_channels])
    # (3) lstm_input.shape =  (16, 10, 396)
    # - 16: batch_size
    # - 10: time steps
    # - 396 = 6 x 6 x 11

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
    # (16, 10, 396) -> (16, 10, 64)
    #
    # (16, 10, 396):
    # 16: batch size
    # 10: time steps
    # 396 = 6 x 6 x 11
    #
    # 16: batch size
    # 10: time steps
    # 64: hidden units
    # 1. for each x_i(t) (example_i's step_t):
    # a (64, 1) = W(64, 396) * x (391, 1) 
    # --------------------------------------------------------------
    # seq_len.shape = (16)
    # An int32/int64 vector sized [batch_size]. 
    # Used to copy-through state and zero-out outputs when past a batch 
    # element's sequence length. So it's more for performance than correctness.
    seq_len = tf.fill([batch_size], feature_w)
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
    
    # (3) lstm_input.shape = (16, 10, 396)
    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_input, sequence_length=seq_len, initial_state=initial_state, dtype=tf.float32, time_major=False)
    
    # (4) outputs.shape = (16, 10, 64)
    # - cell:
    # - param lstm_input: shape = (16, 10, 396) 
    # Edwinn's codes: [16: batch_size, 6: width (after average pooling), 64: height (after average pooling) x channels]
    # should be: [16: batch_size, 16: max_time, 396: depth] (see the param 'time_major')
    # 
    # - param seq_len: shape = (16, )
    # - param initial_state:
    # - param time_major: False.
    # The shape format of the inputs and outputs Tensors. 
    # If true, these Tensors must be shaped [max_time, batch_size, depth].
    # If false, these Tensors must be shaped [batch_size, max_time, depth].
    # Using time_major = True is a bit more efficient because it avoids 
    # transposes at the beginning and end of the RNN calculation. 
    # However, most TensorFlow data is batch-major, 
    # so by default this function accepts input and emits output in batch-major form.
    # 
    # - return outputs: 
    # If time_major == False (default), this will be 
    # a Tensor shaped: [16: batch_size, 10: max_time, 64: cell.output_size].
    # Edwinn's codes: [16: batch_size, 6: width (after average pooling), 64: num_hidden]
    #   Note that the output size = num_hidden because the output of LSTM
    #   is 'a' (a = g(Waa * a + Wax * x + ba)) per se. Therefore, the 
    #   dimension of a is the dimension of the output.
    # should be: [16: batch_size, 10: max_time, 64: cell.output_size]
    
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
    outputs = outputs[:,-1,:]
    # (5) output.shape = (16, 64)
    

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
     
    # --------------------------------------------------------------
    # Paper:
    # (16, 64) -> (16, 4)
    # 1. For each batch, feed the output to a linear layer, W : (4, 64)
    # 2. the output from the linear layer should be (batch = 16, num_classes = 4)
    #
    # the linear layer (y = xW + b)
    # 1. For '(16, 64) -> (16, 4)':
    # 16 x 64 (num_hidden) -> 16 x 4 (num_classes)
    # by a linear layer (y = xW + b)
    # - x (16, 64)
    # - W (64, 4)
    # - b (4, )
    # - y (16, 4)
    # --------------------------------------
    
    # No need to reshape here
    ## (4) outputs.shape = (16, 64)
    # outputs = tf.reshape(outputs, [-1, num_hidden])
    ## (5) output.shape = (16, 4)
    
    # W.shape = (64, 4)
    # b.shape = (4, )
    W = tf.get_variable(name='W_out', shape=[num_hidden, num_classes], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
    b = tf.get_variable(name='b_out', shape=[num_classes], dtype=tf.float32, initializer=tf.constant_initializer())
    # pdb.set_trace()

    #Linear output
    # (5) output.shape = (16, 64)
    linear_output = tf.matmul(outputs, W) + b
    # (6) linear_outputshape = (16, 4)

    return linear_output

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
    '''
    :param input_tensor: 
    :param n: the number of layers in the resnet
    :param num_classes: 
    :param reuse: ?
    :param train:  
    :return layers[-1]: "logits" is the output of the charnet (including ResNET and LSTM) 
    # and is the input for a softmax layer 
    '''
    # pdb.set_trace()
    layers = []
       
    #Append the input tensor as first layer
    # input_tensor.shape = (16, 12, 12, 11)
    layers.append(input_tensor)
    
    # resnet_output_channels = 32 (as in the paper. we currently use
    # 11, MAZE_DEPTH, to enable addition with the residual connection.)
    
    #Add n residual layers
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            # layers[-1] = intput_tensor = (16, 12, 12, 11)
            # block.shape = (16, 12, 12, 11)
            block = residual_block(layers[-1], MAZE_DEPTH) #resnet_output_channels) 
            activation_summary(block)
            layers.append(block)
    
    #Add average pooling
    with tf.variable_scope('average_pooling', reuse=reuse):
        # block.shape = (16, 12, 12, 11) 
        # avg_pool.shape = (16, 6, 6, 11)
        avg_pool = average_pooling_layer(block)
        layers.append(avg_pool)
    
    #Add LSTM layer
    with tf.variable_scope('LSTM', reuse=reuse):
        # layers[-1].shape = avg_pool.shape = (16, 6, 6, 11)
        # 16: Tx
        # 6, 6, 11: the output width, height, and channels from average pooling
        
        # lstm.shape = (16, 6, 4)
        # 16: Ty
        # 6: ?
        # 4: 
        lstm = lstm_layer(layers[-1], train, num_classes)
        layers.append(lstm)        

    #Fully connected
    with tf.variable_scope('fc', reuse=reuse):

        # tf.reduce_mean: Computes the mean of elements across dimensions of a tensor.
        # - param input_tensor: the output from the previous LSTM layer
        # - param axis: The dimensions to reduce
        
        # global average pooling:
        # average across the second axis:
        # from (16, 6, 4) to (16, 4) [6 2-d arrays reduce to 1 2-d array]
        # global_pool.shape = (16, 4)
        global_pool = tf.reduce_mean(layers[-1], [1])
        assert global_pool.get_shape().as_list()[-1:] == [num_classes]

        # def output_layer(input_layer, num_labels):
        # '''
        # :param input_layer: 2D tensor
        # :param num_labels: int. How many output labels in total?
        # :return: output layer Y = WX + B
        # '''
        
        # output.shape = (16, 4)
        output = output_layer(global_pool, num_classes)
        layers.append(output)

    return layers[-1]

def build_pred_head(self):
    pass