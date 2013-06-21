import numpy as np
import scipy.io as sio
import bp,containers

inputs = np.array(([0,0],[0,1],[1,0],[1,1]))        
outputs = np.array(([0,1,1,1]))#.reshape(1,4)
layers = np.array(([2,2,2]))
bpNet = containers.NN(inputs,outputs,layers)
print bpNet
j = bp.trainBp(bpNet,0,50)
feedFwd = bp.feedFwdBp(bpNet,inputs) 
print feedFwd
error = np.sum(feedFwd - outputs)
print 'OR Errors: %d' %(error)

outputs = np.array(([0,0,0,1]))
bpNet = containers.NN(inputs,outputs,layers)
j = bp.trainBp(bpNet,0,50)
feedFwd = bp.feedFwdBp(bpNet,inputs) 
error = np.sum(np.absolute(np.around(feedFwd) - outputs))
print 'AND Errors: %d' %(error)

outputs = np.array(([0,1,1,0]))
bpNet = containers.NN(inputs,outputs,layers)
j = bp.trainBp(bpNet,0,50)
feedFwd = bp.feedFwdBp(bpNet,inputs) 
error = np.sum(np.absolute(np.around(feedFwd) - outputs))
print 'XOR Errors: %d' %(error)

input_layer_size  = 400; 
hidden_layer_size = 25;   
num_labels = 10;          

data = sio.loadmat('ex4data1.mat')
X = data['X']
y = data['y'].reshape(-1) -1

inputs = X
outputs = y
layers = np.array(([400,25,10]))
bpNet = containers.NN(inputs,outputs,layers)
print bpNet
j = bp.trainBp(bpNet,1,50)
feedFwd = bp.feedFwdBp(bpNet,inputs) 
error = np.mean(feedFwd == outputs) * 100.0
print 'Accuracy: %r' %(error)
