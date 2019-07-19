import os
import sys
import glob
import time
from random import shuffle
import numpy as np
import tensorflow as tf
import data_handler as dh

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feat_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def createDataRecord(out_filename, addrs):
    dir = os.getcwd() + '/S002a/'
    handler = dh.DataHandler(dir)
    start = time.time()
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        #Print how many trajectories we have stored every 1000 files
        if not i % 100:
            if 'train' in out_filename:
                print('Generating training data {}/{}'.format(i,len(addrs)))
            elif 'val' in out_filename:
                print('Generating validation data {}/{}'.format(i,len(addrs)))
            else:
                print('Generating testing data {}/{}'.format(i,len(addrs)))
            sys.stdout.flush()
        
        #Load a trajectory and its final goal as label
        #Trajectories are tensors of shape [37x12x12x45]
        #Labels are one hot-encoded vectors with size = range(goals) [38x1]
        traj, label = handler.parse_trajectory(addrs[i])
        flat_traj = traj.flatten()
        #labelbytes = str.encode(''.join(str(e) for e in label))
        #Create a feature
        feature = {
            'steps': _int64_feature(traj.shape[0]),
            'height': _int64_feature(traj.shape[1]),
            'width': _int64_feature(traj.shape[2]),
            'depth': _int64_feature(traj.shape[3]),
            'raw_trajectory': _int64_feat_array(flat_traj),
            'label': _int64_feature(label),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    writer.close()
    sys.stdout.flush()
    end = time.time()
    print('Elapsed time', end-start)

#Import data
addrs = glob.glob('S002a/*')
shuffle(addrs)

#Divide data into 60% train, 20% validation and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]

createDataRecord('train.tfrecord', train_addrs)
createDataRecord('val.tfrecord', val_addrs)
createDataRecord('test.tfrecord', test_addrs)
