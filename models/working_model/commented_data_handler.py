#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class DataHandler(mp.ModelParameter):

The class for parsing txt data.

Note:
  Inherit mp.ModelParameter to share model constants.

@author: Chuang, Yun-Shiuan; Edwinn
"""

import os
import sys
import numpy as np
from random import shuffle
# from commented_utils import plot_trajectory # no longer needs it
import re
import pdb
import commented_model_parameters as mp


class DataHandler(mp.ModelParameter):
    # --------------------------------------
    # Constant block
    # --------------------------------------
    # --------------------------------------
    # Constant: Model parameters
    # --------------------------------------
    # Use inheretance to share the model constants across classes

    def __init__(self):
        #self.find_max_path(dir)
        pass

    def parse_whole_data_set(self, directory, mode, shuf, subset_size = -1, parse_query_state = False):
        '''
        Parse the trajectory (if `parse_query_state = False`) or the query state
        (if `parse_query_state = True`) of a txt file.

        Args:
        :param subset_size: The size of the subset (number of files) to be parsed.
          Default to the special number -1, which means using all the files in
          the directory. When testing the code, this could help reducing the parsing time.
        :param parse_query_state: if 'True', parse only the query states
          and skip the actions; if 'False', parse the whole sequence
          of trajectories

        Returns:
          :train_data:
            the training data.
            if `parse_query_state = False`,
            then return 5D numpy array (num_files, trajectory_size, height, width, depth_trajectory);
            if `parse_query_state = True`,
            then return 4D numpy array (num_files, height, width, depth_query_state);

          :train_labels:
            a batch data. 2D numpy array (num_files, 1)
        '''
        # --------------------------------------
        # List all txt files to be parsed
        # --------------------------------------
        files = self.list_txt_files(directory)

        if subset_size != -1:
          files = files[0:subset_size]

        # --------------------------------------
        # Print out parsing message
        # --------------------------------------
        if not parse_query_state:
          parse_mode = 'trajectories---------------'
        else:
          parse_mode = 'query states---------------'
        print('Parse ', parse_mode)
        print('Found', len(files), 'files in', directory)

        #pdb.set_trace()

        #Shuffle the filenames
        if shuf:
          # note that shuf = False by default
          # See commented_main_model() for details:
          # section: if __name__ == "__main__":
            shuffle(files)
        # pdb.set_trace()

        # Size of data set
        # Train : Vali: Test = 80 : 10 : 10
        train_files = files[0:int(len(files)*0.8)]
        vali_files = files[int(len(files)*0.8):int(len(files)*0.9)]
        test_files = files[int(0.9*len(files)):len(files)]

        #Initialize empty arrays for data
        train_data = []
        train_labels = []
        vali_data = []
        vali_labels =[]
        test_data = []
        test_labels=[]

        if mode == 'train' or mode == 'all':
            print('Parsing training data')
            train_data, train_labels = self.parse_subset(directory, train_files, parse_query_state)
            print('Parsing validation data')
            vali_data, vali_labels = self.parse_subset(directory, vali_files, parse_query_state)

        if mode == 'test' or mode == 'all':
            print('Parsing testing data')
            test_data, test_labels = self.parse_subset(directory, test_files, parse_query_state)

        return train_data, vali_data, test_data, train_labels, vali_labels, test_labels, files, train_files, vali_files, test_files

    def list_txt_files(self,directory):
      '''
        This function wil return all the txt files in a given directory.
        Args:
          :param directory: the directory of the files to be listed.

        Returns:
          :files:
            all the txt files in the directory.
      '''
      # The list is in arbitrary order.
      # (the order has to do with the way the files are indexed on your FileSystem)
      # The order is fixed if runnning from the same machine
      files = os.listdir(directory)
      # pdb.set_trace()
      # Filter out the csv file (only read the txt files)
      r = re.compile(".*.txt")
      files = list(filter(r.match, files))
      return files

    def parse_subset(self, directory, files, parse_query_state, with_label = True):
        '''
        This function wil parse all the files in the directoy and return
        the corresponding tensors and labels.
        Args:
          :param directory:
            the directory of the files to be parse
          :param files:
            the txt files to be parsed
          :param parse_query_state:
            if 'True', parse only the query states
            and skip the actions; if 'False', parse the whole sequence
            of trajectories
          :param with_label:
            whether the query state file contains the final
            target (default to True). Note that this will be False for
            preference inference on equal-distance files since there are no
            final tagrgets in such files.

        Returns:
          :all_data:
            if `parse_query_state == True`,
            return the 4D tensor of the query state
            (num_files, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_QUERY_STATE);
            if `parse_query_state == False`,
            return the 5D tensor of the whole trajectory
            (num_files, trajectory_size, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_TRAJECTORY).
          :all_labels:
            the numeric index of the final target (len(files), 1)
        '''
        # --------------------------------------------------------------
        # Initialize empty arrays and constants
        # --------------------------------------------------------------
        # pdb.set_trace()
        if not parse_query_state:
          all_data = np.empty([self.MAX_TRAJECTORY_SIZE,self.MAZE_WIDTH,self.MAZE_HEIGHT,self.MAZE_DEPTH_TRAJECTORY])
        else:
          all_data = np.empty([self.MAZE_WIDTH,self.MAZE_HEIGHT,self.MAZE_DEPTH_QUERY_STATE])

        all_labels = np.empty([1])

        # local constant
        num_dummy_values =  all_data.shape[0]
        num_files = len(files)
        # --------------------------------------------------------------
        # Parse file one by one
        # --------------------------------------------------------------
        i = 0 # file index
        j = 0 # for tracking progress (%)
        if not parse_query_state:
            # Parse data and labels
            for file in files:
                i += 1
                if i > j*len(files)/100:
                    print('Parsed ' + str(j) + '%')
                    j+=10
                traj, goal = self.parse_trajectory(os.path.join(directory, file))
                all_data = np.vstack((all_data,traj))
                for step in traj:
                    all_labels = np.hstack((all_labels,np.array(goal)))
        else:
            # Parse data and labels
            for file in files:
                i += 1
                if i > j*len(files)/100:
                    print('Parsed ' + str(j) + '%')
                    j+=10
                query_state, goal = self.parse_query_state(os.path.join(directory, file),\
                                                           with_label)
                #pdb.set_trace()
                all_data = np.vstack((all_data,query_state))
                all_labels = np.hstack((all_labels,np.array(goal)))
            #pdb.set_trace()
        print('Parsed ' + str(j) + '%')

        # --------------------------------------------------------------
        # Clean up leading dummy values
        # --------------------------------------------------------------
        # Delete the leading empty elements
        #pdb.set_trace()
        for i in range(num_dummy_values):
            all_data = np.delete(all_data,(0), axis=0)
          # Delete the leading 1 number(see "all_labels = np.empty([1])")
        all_labels = np.delete(all_labels,(0), axis=0)
        #pdb.set_trace()

        # --------------------------------------------------------------
        # Reshaping the data tensor and labels tensor:
        # data = (num_files x dim_1, dim_2, ...) ->
        # data = (num_files, dim_1, dim_2, ...)
        # --------------------------------------------------------------
        if not parse_query_state:
          all_data = all_data.reshape(num_files, self.MAX_TRAJECTORY_SIZE,
                                      self.MAZE_WIDTH,self.MAZE_HEIGHT,
                                      self.MAZE_DEPTH_TRAJECTORY)
          # --------------------------------------------------------------
          # Only retain unique labels
          # test_labels = （total_steps, ） ->
          # test_labels = (num_files, )
          # --------------------------------------------------------------
          all_labels = all_labels[0:-1:self.MAX_TRAJECTORY_SIZE]
        else:
           all_data = all_data.reshape(num_files,
                                      self.MAZE_WIDTH,self.MAZE_HEIGHT,
                                      self.MAZE_DEPTH_QUERY_STATE)
        print('Got ' + str(all_data.shape) + ' datapoints')
        # pdb.set_trace()
        return all_data, all_labels


    def parse_trajectory(self, filename):
        '''
        This function wil return a 4-dim tensor with all the steps in a trajectory defined in the map of the given txt file.
        The tensor will be of shape (MAX_TRAJECTORY_SIZE, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_QUERY_STATE).
        '''
        #self.parse_query_state(filename)

        steps = []
        #output.shape(12, 12, 11, 10)
        output = np.zeros((self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY, self.MAX_TRAJECTORY_SIZE))
        label = ''
        with open(filename) as fp:
            lines = list(fp)
            maze = lines[2:14]

            #Parse maze to 2d array, remove walls.
            i=0
            while i < 12: # in the txt file, each maze has 12 lines
                maze[i]= list(maze[i])
                maze[i].pop(0)
                maze[i].pop(len(maze[i])-1)
                maze[i].pop(len(maze[i])-1)
                i+=1

            #Original maze (without walls)
            np_maze = np.array(maze)

            #Plane for obstacles
            np_obstacles = np.where(np_maze == '#', 1, 0).astype(np.int8)

            #Plane for agent's initial position
            np_agent = np.where(np_maze == 'S', 1, 0).astype(np.int8)

            #Planes for each possible goal
            #targets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m']
            targets = ['C','D','E','F'] # for the simplified 4-targets mazes
            np_targets = np.repeat(np_maze[:, :, np.newaxis], len(targets), axis=2)
            for target, i in zip(targets, range(len(targets))):
                np_targets[:,:,i] = np.where(np_maze == target, 1, 0)
            # np_targets.shape = (12, 12, 4)
            np_targets = np_targets.astype(int)

            #Parse trajectory into 2d array
            trajectory = lines[15:]
            agent_locations = []
            for i in trajectory:
                i = i[1:len(i)-2]
                tmp = i.split(",")
                try:
                    agent_locations.append([tmp[0],tmp[1]])
                except:pass
            possible_actions=['right', 'left', 'up', 'down', 'goal']
            for i in range(len(agent_locations) - 1):

                #Create a 12x12sx4 tensor for each t(x,a)
                np_actions = np.zeros((12,12,len(possible_actions)), dtype=np.int8)
                #Determine the type of action
                if agent_locations[i][0] > agent_locations[i+1][0]:
                    layer = 'left'
                elif agent_locations[i][0] < agent_locations[i+1][0]:
                    layer = 'right'
                elif agent_locations[i][1] > agent_locations[i+1][1]:
                    layer = 'up'
                elif agent_locations[i][1] < agent_locations[i+1][1]:
                    layer = 'down'
                # Assign a value of 1 to the location where the action starts
                # np_actions.shape = (12, 12, 5)
                np_actions[int(agent_locations[i][1])-1, int(agent_locations[i][0])-1, possible_actions.index(layer)] = 1

                # For each step:
                # np_tensor.shape = (12, 12, 11)
                # DEPTH = 11 layers = 1 (obstacle) + 4 (targets) + 1 (agent initial position) + 5 (actions)
                np_tensor = np.dstack((np_obstacles,np_targets,np_agent,np_actions))
                steps.append(np_tensor)
                output = np.array(steps)

            #The last tensor of every trajectory will be the position of the goal.
            np_actions = np.zeros((12,12,len(possible_actions)), dtype=np.int8)
            np_actions[int(agent_locations[-1][1])-1, int(agent_locations[-1][0])-1, possible_actions.index('goal')] = 1

            # For the last step:
            # np_tensor.shape = (12, 12, 11)
            # DEPTH = 11 layers = 1 (obstacle) + 4 (targets) + 1 (agent initial position) + 5 (actions)
            np_tensor = np.dstack((np_obstacles,np_targets,np_agent,np_actions))

            # ---------------------------------------
            # Generate the label (could be training_label, valid_label, testing_label)
            # (y, an int from 0 to 3)
            # ---------------------------------------
            # Make the label from the letter in the final position of the agent
            goal = np_maze[int(agent_locations[-1][1])-1][int(agent_locations[-1][0])-1]
            char_to_int = dict((c, i) for i, c in enumerate(targets))
            integer_encoded = char_to_int[goal]
            #label = [0 for _ in range(len(targets))]
            #label[integer_encoded] = 1

            #Return label as a number
            label = int(integer_encoded)

            # ---------------------------------------
            # Generate each training/valid/testing example (one trajectory)
            # ---------------------------------------
            # Put everything step together in a 4-dim tensor: "output" (one trajectory)
            # each output (4-dim tensor, will be of shape after padding/truncating = (10, 12, 12, 11))
            # contain 10 layers ("np_tensor", 3-dim tensor, shape = (12, 12, 11))
            steps.append(np_tensor)
            output = np.array(steps)

            # pdb.set_trace()
            pad_size = int(self.MAX_TRAJECTORY_SIZE - output.shape[0])

            #Zeroes pre-padding to max length
            if pad_size > 0:
                np_pad = np.zeros((self.MAZE_HEIGHT,self.MAZE_WIDTH,self.MAZE_DEPTH_TRAJECTORY), dtype=np.int8)
                for i in range(pad_size):
                    # insert the zero layer to the head
                    output = np.insert(output, 0, np_pad, axis=0)

            #Truncating trajectory to max length
            elif pad_size < 0:
                for i in range(abs(pad_size)):
                    # remove the first step
                    output = np.delete(output, 0, axis=0)

        fp.close()
        return output, label

    def parse_query_state(self, filename, with_label = True):
        '''
        This function wil return a 3-dim tensor including the static information
        of a maze, including 6 layers:
          (1) obstacles (1 layer)
          (2) agent initial position (1 layer)
          (3)target positions (4 layers)
        This function is primary for prednet as it needs query state as input.

        Args:
          :param filename:
            the txt file of the trajectory of interest
          :param with_label:
            whether the query state file contains the final target (default to True).
            Note that this will be False for preference inference on
            equal-distance files since there are no final tagrgets in such files.

        Returns:
          :np_query_state_tensor:
            a batch data. 3D numpy array of the query state
            (MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH_QUERY_STATE)
          :label: the numeric index of the final target. This will be -1
          if `with_label = False`
        '''
        # --------------------------------------------------------------
        # Construct the query state tensor: (12, 12, 6)
        # --------------------------------------------------------------
        #output.shape(12, 12, 6)
        # pdb.set_trace()
        with open(filename) as fp:
            lines = list(fp)
            maze = lines[2:14]

            #Parse maze to 2d array, remove walls.
            i=0
            while i < 12: # in the txt file, each maze has 12 lines
                maze[i]= list(maze[i])
                maze[i].pop(0)
                maze[i].pop(len(maze[i])-1)
                maze[i].pop(len(maze[i])-1)
                i+=1

            #Original maze (without walls)
            np_maze = np.array(maze)

            #Plane for obstacles
            np_obstacles = np.where(np_maze == '#', 1, 0).astype(np.int8)

            #Plane for agent's initial position
            np_agent = np.where(np_maze == 'S', 1, 0).astype(np.int8)

            #Planes for each possible goal
            #targets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m']
            targets = ['C','D','E','F'] # for the simplified 4-targets mazes
            np_targets = np.repeat(np_maze[:, :, np.newaxis], len(targets), axis=2)
            for target, i in zip(targets, range(len(targets))):
                np_targets[:,:,i] = np.where(np_maze == target, 1, 0)
            # np_targets.shape = (12, 12, 4)
            np_targets = np_targets.astype(int)

            # For each trajectory:
            # np_query_state_tensor.shape = (12, 12, 6)
            # DEPTH = 11 layers = 1 (obstacle) + 4 (targets) + 1 (agent initial position)
            np_query_state_tensor = np.dstack((np_obstacles,np_targets,np_agent))

            if with_label:
              # --------------------------------------------------------------
              # Retrieve the final target (could be training_label, valid_label, testing_label):
              # size = 1, an int from 0 to 3
              # --------------------------------------------------------------
              #Parse trajectory into 2d array
              # pdb.set_trace()
              trajectory = lines[15:]
              agent_locations = []
              for i in trajectory:
                  i = i[1:len(i)-2]
                  tmp = i.split(",")
                  try:
                      agent_locations.append([tmp[0],tmp[1]])
                  except:pass

              # Make the label from the letter in the final position of the agent
              goal = np_maze[int(agent_locations[-1][1])-1][int(agent_locations[-1][0])-1]
              char_to_int = dict((c, i) for i, c in enumerate(targets))
              integer_encoded = char_to_int[goal]
              #label = [0 for _ in range(len(targets))]
              #label[integer_encoded] = 1

              #Return label as a number
              label = int(integer_encoded)
            else:
              # --------------------------------------------------------------
              # When there is no label in the query state file.
              # --------------------------------------------------------------
              label = int(-1)
        fp.close()
        # pdb.set_trace()
        return np_query_state_tensor, label

    def find_max_path(self, dir):
        paths = []
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename)) as fp:
                lines = list(fp)
                trajectory = len(lines[15:-1])
                paths.append(trajectory)
        self.MAX_TRAJECTORY_SIZE = max(paths)
        print("Longest trajectory:", self.MAX_TRAJECTORY_SIZE)
        #NHWC N: Number of images in a batch, H: Height, W: Width, C: Channels


if __name__ == "__main__":
    #This will get the trajectory of the specified file and plot a sequence of images showing the result of the parse.
    dir = os.getcwd() + '/S002a/'
    file = 'S002_1'
    dh = DataHandler(dir)
    dh.parse_trajectories(dir, mode='all', shuf=False)
    #out, label = dh.parse_trajectory(dir + file + '.txt')
    #print(out.shape)
    #print(label)
    #plot_trajectory(file, out)
