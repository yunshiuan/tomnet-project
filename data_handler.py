import os
import sys
import numpy as np
from random import shuffle
from utils import plot_trajectory
import re

class DataHandler(object):

    MAZE_WIDTH = 12
    MAZE_HEIGHT = 12
    MAZE_DEPTH = 11
    MAX_TRAJECTORY_SIZE = 10

    def __init__(self, dir):
        #self.find_max_path(dir)
        pass

    def parse_trajectories(self, directory, mode, shuf):
        
        #Make a trajectory with each step same label
        files = os.listdir(directory)
        # Filter out the csv file (only read the txt files) 
        r = re.compile(".*.txt") 
        files = list(filter(r.match, files)) # Read Note    

        print('Found', len(files), 'files in', directory)
        
        #Shuffle the filenames
        if shuf:
            shuffle(files)

        #Start testing with a 50-25-25 ratio
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
            train_data, train_labels = self.parse_subset(directory, train_files)
            print('Parsing validation data')
            vali_data, vali_labels = self.parse_subset(directory, vali_files)
        
        if mode == 'test' or mode == 'all':
            print('Parsing testing data')
            test_data, test_labels = self.parse_subset(directory, test_files)
        
        return train_data, vali_data, test_data, train_labels, vali_labels, test_labels

    def parse_subset(self, directory, files):
        all_data = np.empty([self.MAX_TRAJECTORY_SIZE,self.MAZE_WIDTH,self.MAZE_HEIGHT,self.MAZE_DEPTH])
        all_labels = np.empty([1])
        
        i = 0
        j = 0
        for file in files:
            i += 1
            if i > j*len(files)/100:
                print('Parsed ' + str(j) + '%')
                j+=10
            traj, goal = self.parse_trajectory(directory + file)
            all_data = np.vstack((all_data,traj))
            for step in traj:
                all_labels = np.hstack((all_labels,np.array(goal)))
        print('Parsed ' + str(j) + '%')
        for i in range(10):
            all_data = np.delete(all_data,(0), axis=0)
        all_labels = np.delete(all_labels,(0), axis=0)
        print('Got ' + str(all_data.shape) + ' datapoints')
        return all_data, all_labels


    def parse_trajectory(self, filename):
        '''
        This function wil return a 4-dim tensor with all the steps in a trajectory defined in the map of the given txt file.
        The tensor will be of shape (MAX_TRAJECTORY_SIZE, MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH).
        '''
        
        steps = []
        output = np.zeros((self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH, self.MAX_TRAJECTORY_SIZE))
        label = ''
        with open(filename) as fp:
            lines = list(fp)
            maze = lines[2:14]
            
            #Parse maze to 2d array, remove walls.
            i=0
            while i < 12:
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
            targets = ['C','D','E','F']
            np_targets = np.repeat(np_maze[:, :, np.newaxis], len(targets), axis=2)
            for target, i in zip(targets, range(len(targets))):
                np_targets[:,:,i] = np.where(np_maze == target, 1, 0)
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
                    layer = 'right'
                elif agent_locations[i][0] < agent_locations[i+1][0]:
                    layer = 'left'
                elif agent_locations[i][1] > agent_locations[i+1][1]:
                    layer = 'up'
                elif agent_locations[i][1] < agent_locations[i+1][1]:
                    layer = 'down'                
                #Assign a value of 1 to that location
                np_actions[int(agent_locations[i][1])-1, int(agent_locations[i][0])-1, possible_actions.index(layer)] = 1
                np_tensor = np.dstack((np_obstacles,np_targets,np_agent,np_actions))
                steps.append(np_tensor)
                output = np.array(steps)
            
            #The last tensor of every series will be the position of the goal.
            np_actions = np.zeros((12,12,len(possible_actions)), dtype=np.int8)
            np_actions[int(agent_locations[-1][1])-1, int(agent_locations[-1][0])-1, possible_actions.index('goal')] = 1
            np_tensor = np.dstack((np_obstacles,np_targets,np_agent,np_actions))

            #Make the label from the letter in the final position of the agent in one hot encoding style
            goal = np_maze[int(agent_locations[-1][1])-1][int(agent_locations[-1][0])-1]
            char_to_int = dict((c, i) for i, c in enumerate(targets))
            integer_encoded = char_to_int[goal]
            #label = [0 for _ in range(len(targets))]
            #label[integer_encoded] = 1
            
            #Return label as a number
            label = int(integer_encoded)
            
            #Put everything together in a 4-dim tensor            
            steps.append(np_tensor)
            output = np.array(steps)

            pad_size = int(self.MAX_TRAJECTORY_SIZE - output.shape[0])
            
            #Zeroes pre-padding to max length
            if pad_size > 0:
                np_pad = np.zeros((self.MAZE_HEIGHT,self.MAZE_WIDTH,self.MAZE_DEPTH), dtype=np.int8)
                for i in range(pad_size):
                    output = np.insert(output, 0, np_pad, axis=0)
            
            #Truncating trajectory to max length
            elif pad_size < 0:
                for i in range(abs(pad_size)):
                    output = np.delete(output, 0, axis=0)
        
        fp.close()
        return output, label

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
