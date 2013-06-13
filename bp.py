import numpy as np
#import os 
#import sys
#import matplotlib.pyplot as plt
#import scipy.optimize as sopt
import scipy.io as sio
#import time 

class memObj:
    def __init__(self, obj): self.obj = obj
    def get(self):    return self.obj
    def set(self, obj):      self.obj = obj

class backProp:
    EPSILON = 0.01
    
    def sigmoid(self,z):
        g = 1 /(1 + np.exp(-z))
        return g


    def randInit(self,lIn, lOut):
        return ((np.random.rand(lOut,lIn + 1) * 2.0 * self.EPSILON) - self.EPSILON).reshape(-1)

    def getTheta(self,tIn,lPos):
        t = tIn[self.layers[lPos,2]:self.layers[lPos,0]*self.layers[lPos,1]+self.layers[lPos,2]]
        return t.reshape(self.layers[lPos,0],self.layers[lPos,1])

    def hypothesis(self,t,XIn):
	z = []
	a = []
	#print XIn.shape
        nOnes = np.ones((XIn.shape[0],1))
        #print nOnes.shape
        m_a = memObj(np.hstack((nOnes,XIn)))
	a.append(m_a)
	z.append(0)#print a
        for i in range(1,self.l.size):
   	    tL = self.getTheta(t,i-1)
   	    a_i_1 = a[i-1].get()
 #  	    print i-1
 #  	    print a_i_1.shape,tL.T.shape
   	    m_z = memObj(np.dot(a_i_1,tL.T))
            z.append(m_z)
            nOnes = np.ones((a_i_1.shape[0],1))
            m_z = z[i].get()
            sig_m_z = self.sigmoid(m_z)
            m_a = memObj(np.hstack((nOnes,sig_m_z)))
#            print sig_m_z.shape
            a.append(m_a)
#        print a
#       print self.l.size
        h = a[self.l.size-1].get()[:,1:]
#        print h.shape
        return h
                                        
    def costFunction(self,t,XIn):
	z = []
	a = []
        m = XIn.shape[0]
        nOnes = np.ones((XIn.shape[0],1))
        m_a = memObj(np.hstack((nOnes,XIn)))
	a.append(m_a)
	z.append(0)#print a
        for i in range(1,self.l.size):
   	    tL = self.getTheta(t,i-1)
   	    a_i_1 = a[i-1].get()
   	    m_z = memObj(np.dot(a_i_1,tL.T))
            z.append(m_z)
            nOnes = np.ones((a_i_1.shape[0],1))
            m_z = z[i].get()
            sig_m_z = self.sigmoid(m_z)
            m_a = memObj(np.hstack((nOnes,sig_m_z)))
            a.append(m_a)
        h = a[self.l.size-1].get()[:,1:]
        temp1 = (1.0 - self.y).T * np.log(1.0 - h)
	temp2 = -1.0 * self.y.T * np.log(h)
	temp3 = temp2 - temp1 
	J_unreg = 1.0/m * np.sum(np.sum(temp3, axis=1))
        return J_unreg

    def __str__(self):
        return 'Inputs (%d,%d)\nOutputs (%d,%d)\nlayers:\n%s' %(self.X.shape[0],self.X.shape[1],self.y.shape[0],self.y.shape[1],str(self.l))

    def __init__(self,inputs,outputs,layers):
        self.X = inputs
        self.y = outputs
        self.l = layers
        self.lcount = layers.size
        self.theta = self.randInit(layers[0],layers[1])
        self.layers = np.array((layers[1],layers[0]+1,0))
        for i in range(1,layers.size-1):
            pos = self.theta.size
            self.theta = np.hstack((self.theta,self.randInit(layers[i], layers[i + 1])))
            self.layers = np.vstack((self.layers, np.array((layers[i+1],layers[i]+1,pos))))
        
        print self.layers
#        print self.theta.size

if __name__ == "__main__":
    inputs = np.array(([0,0],[0,1],[1,0],[1,1]))        
    outputs = np.array(([0,1,1,0])).reshape(4,1)
    layers = np.array(([2,2,2,2,1]))
    bp = backProp(inputs,outputs,layers)
    print bp
#    print bp.getTheta(0)
#    print bp.getTheta(1)
#    print bp.getTheta(2)
#    print bp.getTheta(3)
    
    h = bp.hypothesis(bp.theta,bp.X)
    print h
    j = bp.costFunction(bp.theta,bp.X)
    print j
    