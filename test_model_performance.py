import sys
sys.path.insert(0, '/Users/vimchiz/bitbucket_local/observer_model_group')
import resnet as rn
import data_handler as dh  
import main_model as mm

# Constants
model = mm.Model()
model.test()
