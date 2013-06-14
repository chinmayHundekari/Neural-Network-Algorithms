"""
Neural network with Backpropagation
===================================

Provides
  1. An implementation of backpropgation algorithm.
  2. A class to store, train and execute a neural network.
  
"""


import numpy as np
#import matplotlib.pyplot as plt
import scipy.optimize as sopt
#import scipy.io as sio
import time 

__author__ = "Chinmay Hundekari"
__version__ = "0.0.1"
__status__ = "Development"


class memObj:
    """A wrapper for objects.

        This wrapper is used to store multi-size arrays in a single list by
        backProp.
    """
    def __init__(self, object):
        self.obj = object
        
    def get(self):
        return self.obj
        
    def set(self, object):
        self.obj = object
        

class backProp:
    """A class to train, run and store a neural network based on 
    backpropagation.

    """
    EPSILON = 0.1
    
    def time_end(self,startTime):
        return (time.clock() - startTime)
        
    def sigmoid(self,z):
        g = 1 /(1 + np.exp(-z))
        return g

    def memArray(self,obj,size):
        objArr = []
        for i in range(0,size):
            objArr.append(memObj(obj))
        return objArr

    def randInit(self,lIn, lOut):
        return ((np.random.rand(lOut,lIn + 1) * 2.0 * self.EPSILON) - self.EPSILON).reshape(-1)

    def getThetaLayer(self,tIn,lPos):
        t = tIn[self.layers[lPos,2]:self.layers[lPos,0]*self.layers[lPos,1]+self.layers[lPos,2]]
        return t.reshape(self.layers[lPos,0],self.layers[lPos,1])

    def feedForward(self, X):
        """Executes the initialized neural network based on the existing values
        of theta
        
        Args:
            X: input vectors for which the neural network prediction 
            has to be calculated.
            
                Format : numpy array of shape n * m, where 
                n = number of inputs
                m = number of inputs in the input vector
            
        Returns:
            Values predicted by the neural network.
            
                Format: numpy array of shape n * k, where
                k = number of output features to be predicted.
        """
        Xiter = X
        mOnes = np.ones((Xiter.shape[0],1))
        for i in range(0, self.l.size-1):
            Xiter = self.sigmoid(np.dot(np.hstack((mOnes,Xiter)),self.getThetaLayer(self.theta,i).T))
        return Xiter.T
        
    def costGradient(self,t):
        z = []
        a = []
        m = self.X.shape[0]
        nOnes = np.ones((self.X.shape[0],1))
        m_a = memObj(np.hstack((nOnes,self.X)))
        a.append(m_a)
        z.append(0)

        for i in range(1,self.l.size):
            tL = self.getThetaLayer(t,i-1)
            a_i_1 = a[i-1].get()
            m_z = memObj(np.dot(a_i_1,tL.T))
            z.append(m_z)
            nOnes = np.ones((a_i_1.shape[0],1))
            m_z = z[i].get()
            sig_m_z = self.sigmoid(m_z)
            m_a = memObj(np.hstack((nOnes,sig_m_z)))
            a.append(m_a)
        h = a[self.l.size-1].get()[:,1:]

        delta = self.memArray(0,self.l.size)
        D = self.memArray(0,self.l.size-1)
        m_delta = h - self.y.T
        delta[self.l.size-1].set(m_delta)
        m_D = np.dot(a[self.l.size-2].get().T,delta[self.l.size-1].get())
        D[self.l.size-2].set(m_D)
 
        for i in range(self.l.size-3,-1,-1):
            tL = self.getThetaLayer(t,i+1)
            m_delta = np.dot(delta[i+2].get(), tL) * (a[i+1].get() * (1 - a[i+1].get()))
            m_delta = m_delta[:,1:]
            delta[i+1].set(m_delta)
            m_D = np.dot(a[i].get().T, delta[i+1].get())
            D[i].set(m_D)
        
        grad = (D[0].get()/m)
        grad = grad.reshape(-1, order='F')
        for i in range(1, self.l.size-1):
            grad = np.hstack((grad, (D[i].get()/m).reshape(-1, order='F')))

        return grad
        
    def costFunction(self,t):
        z = []
        a = []
        m = self.X.shape[0]
        nOnes = np.ones((self.X.shape[0],1))
        m_a = memObj(np.hstack((nOnes,self.X)))
        a.append(m_a)
        z.append(0)
        for i in range(1,self.l.size):
            tL = self.getThetaLayer(t,i-1)
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

    def trainNet(self,lamda, maxIter):
        """Train the initilized neural network by backpropogation. 

        Args:            
            lamda: Regularisation parameter.
                        
            maxIter: Maximum iterations before stopping training.

        Returns:
            Final cost of the function.

        """
        t = self.theta
        startTime = time.clock()
        self.theta = sopt.fmin_cg(bp.costFunction,t,bp.costGradient,(),maxiter=maxIter)
        endTime = self.time_end(startTime)
        J = self.costFunction(self.theta)
        print "Time taken to train the network is ",endTime,"sec. with the final cost at ", J
        return J

    def setTheta(self,theta):
        """A function to forcefully set the value of the weights instead of 
        the present randomly initialized values.
        
        Args:
            theta: the complete theta matrix
        """
        self.theta = theta.reshape(-1)
        
    def getTheta(self):
        """A function to obtain the values of the weight matrix.
        
        Returns:
            the complete weight matrix as a 1-D array.
        """
        return self.theta
        

    def __str__(self):
        return 'Inputs (%d,%d)\nOutputs (%d,%d)\nlayers %s' %(self.X.shape[0],self.X.shape[1],self.y.shape[0],self.y.shape[1],str(self.l))

    def __init__(self,inputs,outputs,layers):
        """Initialize the neural network with inputs, outputs and 
        form the base architecture of the network.

        Args:
            
            inputs: A 2-dimensional array where each row is an input vector
            and each column an input variable of the vector.
            
                Format : numpy array of shape n * m, where 
                n = number of input data points
                m = number of inputs in the input vector
        
            outputs: A 2-dimensional array where each row is an output 
            vector and each column an output variable of the vector.
            
                Format : numpy array of shape k * n, where 
                k = number of features in the output vector
                n = number of outputs
        
            layers: A 1-dimensional array where each value specifies the
            number of neurons in the layer. This includes the input
            layer, the hidden layers and an output layer. The input
            layer must assume 1 neuron per input.
            
                Format : 1D numpy array, where 
                layers[0] = m = number of inputs
                layers[1...l] = hidden layers with value equal to number of 
                neurons in each layer.
                layers[l+1] = k = number of features in the output vector

            Returns:
            NIL.
        """
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

if __name__ == "__main__":
    inputs = np.array(([0,0],[0,1],[1,0],[1,1]))        
    outputs = np.array(([1,0,1,0])).reshape(1,4)
    layers = np.array(([2,2,1]))
    bp = backProp(inputs,outputs,layers)
    print bp
    j = bp.trainNet(0,50)
    feedFwd = bp.feedForward(inputs) 
    error = np.sum(np.absolute(np.around(feedFwd) - outputs))
    print np.absolute(np.around(feedFwd))
    print 'Errors: %d' %(error)
    
    
