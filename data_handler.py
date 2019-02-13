import os
import sys
import numpy as np
from utils import plot_trajectory

class DataHandler(object):

    MAZE_WIDTH = 12
    MAZE_HEIGHT = 12
    MAZE_DEPTH = 45
    MAX_TRAJECTORY_SIZE = 10

    def __init__(self, dir):
        #self.find_max_path(dir)
        pass

    def parse_all_trajectories(self, directory):
        #Make a trajectory with each step same label
        print('Loading data from txt files...')
        files = os.listdir(directory)
        all_data = np.empty([10,12,12,45])
        all_labels = np.empty([1])
        for file in files:
            #print('parsing file', file)
            traj, goal = self.parse_trajectory(directory + file)
            all_data = np.vstack((all_data,traj))
            for step in traj:
                all_labels = np.hstack((all_labels,np.array(goal)))
        for i in range(10):
            all_data = np.delete(all_data,(0), axis=0)
        all_labels = np.delete(all_labels,(0), axis=0)
        #print(all_labels)
        print('Got data points of shape ' + str(all_data.shape))
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
            targets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m']
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
                np_pad = np.zeros((12,12,45), dtype=np.int8)
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
    dir = os.getcwd() + '/S001a/'
    file = 'S001_1'
    dh = DataHandler(dir)
    data, labels = dh.parse_all_trajectories(dir)
    #out, label = dh.parse_trajectory(dir + file + '.txt')
    #print(out.shape)
    #print(label)
    #plot_trajectory(file, out)
