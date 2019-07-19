import os
import sys
import glob
from random import shuffle
import numpy as np
import tensorflow as tf
from utils import plot_trajectory

class TFRecordExtractor:
    MAZE_WIDTH = 12
    MAZE_HEIGHT = 12
    MAZE_DEPTH = 11
    MAX_TRAJECTORY_SIZE = 10
    
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)
        
    def extract_fn(self, tfrecord):
        #Note: If we change the number of goals the numbers in raw trajectory and label must be changed accordingly
        features = {
            'steps': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'raw_trajectory': tf.FixedLenFeature([self.MAZE_WIDTH*self.MAZE_HEIGHT*self.MAZE_DEPTH*self.MAX_TRAJECTORY_SIZE], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        sample = tf.parse_single_example(tfrecord, features)
        
        traj = sample['raw_trajectory']
        traj_shape = tf.stack([sample['steps'], sample['height'], sample['width'], sample['depth']])
        shaped_traj = tf.reshape(traj, traj_shape)
        label = sample['label']
        
        return shaped_traj, label
    
    #Use this function to verify that the shapes of the trajs and labels are right
    def recreate_traj(self):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self.extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_traj = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    traj, label = sess.run(next_traj)
                    print(traj.shape, label)
            except:
                pass

    def input_fn(self, filenames, train, batch_size=32, buffer_size=2048):
        dataset = tf.data.TFRecordDataset(filenames=filenames)
        dataset = dataset.map(self.extract_fn)

        if train:
            dataset = dataset.shuffle(buffer_size=buffer_size)
            num_repeat = None
        else:
            num_repeat = 1

        dataset = dataset.repeat(num_repeat)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        traj_batch, label_batch = iterator.get_next()

        return traj_batch, label_batch

    def train_input_fn(self):
        return self.input_fn(filenames=['train.tfrecord','test.tfrecord'], train=True)
    
    def val_input_fn(self):
        return self.input_fn(filenames=['val.tfrecord'], train=False)

if __name__ == '__main__':
    t = TFRecordExtractor('test.tfrecord')
    t.recreate_traj()