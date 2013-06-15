import numpy as np
import bp,containers

inputs = np.array(([0,0],[0,1],[1,0],[1,1]))        
outputs = np.array(([1,0,1,0])).reshape(1,4)
layers = np.array(([2,2,1]))
bpNet = containers.NN(inputs,outputs,layers)
print bpNet
j = bp.trainBp(bpNet,0,50)
feedFwd = bp.feedFwdBp(bpNet,inputs) 
error = np.sum(np.absolute(np.around(feedFwd) - outputs))
print np.absolute(np.around(feedFwd))
print 'Errors: %d' %(error)
