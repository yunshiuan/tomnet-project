from os import path
import numpy as np



#Constants
PATH_ROOT='/Users/vimchiz/github_local/observer_model/Social Network Simulation Data (24 agents)'
FILE_TEST = path.join(PATH_ROOT,'S001_18.txt')

# =============================================================================
# #Test the parse_trajectory() function
# =============================================================================

#This function returns an array of n 12x12x(1+1+38+4) tensors, one for each of 
#the n actions encoded in the txt file
MAZE_WIDTH = 12
MAZE_HEIGHT = 12
MAZE_DEPTH = 45
MAX_TRAJECTORY_SIZE = 10
output = np.zeros((MAZE_WIDTH, MAZE_HEIGHT, MAZE_DEPTH, MAX_TRAJECTORY_SIZE))

filename = FILE_TEST
fp = open(filename)
#with open(filename) as fp:
lines = list(fp)
maze = lines[2:14]#crop along the maze

#Parse maze to 2d array, remove walls.
i=0
while i < 12: #this is the length of the square maze
    maze[i]= list(maze[i])
    maze[i].pop(0) #pop the leftmost wall
    maze[i].pop(len(maze[i])-1) #pop the '\n'
    maze[i].pop(len(maze[i])-1) #pop the rightmost wall
    i+=1 #iterate until reaching the botom of the maze

#Original maze (without walls)
np_maze = np.array(maze)

#Plane for obstacles (within the maze) (labeld as true)
np_obstacles = np.where(np_maze == '#', 1, 0).astype(bool)

#Plane for agent's initial position (labeled as true)
np_agent = np.where(np_maze == 'S', 1, 0).astype(bool)

#Planes for each possible goal (38 targets)
targets = ['A','B','C','D','E','F','G','H','I','J','K',
           'L','M','N','O','P','Q','R','T','U','V','W',
           'X','Y','Z','a','b','c','d','e','f','g','h',
           'i','j','k','l','m']
np_targets = np.repeat(np_maze[:, :, np.newaxis], len(targets), axis=2)
# =============================================================================
# Note
# =============================================================================
# # The usage of np.newaxis
# np.shape(np_maze[:, :])
# -> (12,12)
# np.shape(np_maze[:, :, np.newaxis])
# -> (12,12,1)
# =============================================================================
# # np.repeat(np_maze[:, :, np.newaxis], len(targets), axis=2)
# -> Repeat the np_maze[:, :, np.newaxis] for 38 times,
# and concatenate the repeats alon the 2nd dimension
# -> This results in a array with the shape of (12,12,38).
# Each of the 38 elements is a 12 x 12 maze.
# =============================================================================



for target, i in zip(targets, range(len(targets))):
    np_targets[:,:,i] = np.where(np_maze == target, 1, 0)
np_targets = np_targets.astype(bool)
# =============================================================================
# Note
# =============================================================================
## range(number) is a series of number from 0 to number-1
# e.g., range(2) is 0, 1
# =============================================================================
# # zip(): How to iterate through two lists in parallel? 
# for x, y in zip(a, b):
# -> x is from a, y is from b
# =============================================================================



# =============================================================================
# #Parse trajectory into 2d array
# =============================================================================
# get the trajectory composed of the x,y coordiante of each move
trajectory = lines[15:]
agent_locations = []
for i in trajectory:
    i = i[1:len(i)-2] #get rid of the "\n"
    tmp = i.split(",") #get the x and y coordinare
    try:
        agent_locations.append([tmp[0],tmp[1]])
    except:pass
# convert the positions into actions
possible_actions=['right', 'left', 'up', 'down']
for i in range(len(agent_locations) - 1):
    
    #Create a 12x12sx4 tensor for each t(x,a)
    #This create an array full of "False" (shape: 12x12x4)
    np_actions = np.zeros((12,12,len(possible_actions)), dtype=bool)

    #Determine the type of action
    if agent_locations[i][0] > agent_locations[i+1][0]: 
        layer = 'right' #should be left
    if agent_locations[i][0] < agent_locations[i+1][0]: 
        layer = 'left' #should be right
    if agent_locations[i][1] > agent_locations[i+1][1]:
        layer = 'up'
    if agent_locations[i][1] < agent_locations[i+1][1]:
        layer = 'down'
    
    #Assign a value of 1 to that location
    np_actions[int(agent_locations[i][1])-1, int(agent_locations[i][0])-1,
               possible_actions.index(layer)] = True

    #Put everything toegether
    np_tensor = np.dstack((np_obstacles,np_targets,np_agent,np_actions))
#    output.append(np_tensor)
    
    ## Output shape: nx12x12x(1+1+38+4)
    # -> np.array(output).shape 
    # -> (2, 12, 12, 44)
    #n: number of actions (2 in this case with 3 positions)
    #12x12: maze size
    #1:obstacles
    #1:targets
    #38: agents
    #4:actions

fp.close()

#Check result
np_output = np.array(output)
np_agent_locations = np.array(agent_locations).astype(int)
n_action = len(np_agent_locations)-1

#lines
print('Agent location: \n' + 
      np.array2string(np_agent_locations))

for step in range(n_action):
    action_location_x = np_agent_locations[step][0] - 1
    action_location_y = np_agent_locations[step][1] - 1
    action_array = np_output[step,
                             action_location_y, #the order of y and x, see line 112
                             action_location_x,
                             [40,41,42,43]]
    action_index = np.where(action_array)[0][0]
    action_name = possible_actions[action_index]
    print('The step ' + str(step+1) + ' is ' + action_name)

# np.output[0,5,5,[40,41,42,43]] #down
# np.output[1,6,5,[40,41,42,43]] #left (Should be right!)