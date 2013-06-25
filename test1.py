import numpy as np
import scipy.io as sio
import bp,containers

# First run DataProcess.py
data = sio.loadmat('bollinger_inputs.mat')
#print data
X = data['X']
y = data['y'].reshape(-1)
np.set_printoptions(threshold='nan')
print X.shape   
print y.shape
print np.sum(y>0.5)

inputs = X
inp_vec = X.shape[1]
outputs = y > 0.5
layers = np.array(([inp_vec,5,5,2]))
bpNet = containers.NN(inputs,outputs,layers)
print bpNet
j = bp.trainBp(bpNet,1,50)
bp.boolAnalytics(bpNet,inputs,outputs) 
