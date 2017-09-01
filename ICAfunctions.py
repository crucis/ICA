import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.cluster.vq import whiten
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats import entropy, chi2_contingency, shapiro

def g(y):
    return np.multiply(np.power(y,2),np.sign(y))
    #return np.tanh(10*y)

def f(y):
    return np.power(y,3)

def NPCA_RLS(mixtures, runs = 5):
    P = np.identity(mixtures.shape[0])
    W = np.identity(mixtures.shape[0])
    #dW = W
    y = np.zeros(mixtures.shape)
    beta = 0.9
    whitenedMixtures = whiten(mixtures)
    
    for j in np.arange(runs):
        for i in np.arange(whitenedMixtures.shape[1]):
            y[:,i] = np.dot(W, whitenedMixtures[:,i])
            z = g(y[:,i])
            h = np.dot(P,z)
            m = h/(beta + np.dot(z.T, h))

            Triangle = P - np.outer(m, h.T)
            lowerIndices = np.tril_indices(whitenedMixtures.shape[0])
            Triangle[lowerIndices] = Triangle.T[lowerIndices]

            P = (1/beta) * Triangle
            e =  whitenedMixtures[:,i] - np.dot(W.T,z)
            
            dW = np.outer(m, e.T)
            
            W = W + dW
            if (np.isnan(W).any() == True):
                print('Lost convergence at iterator %d'%i)
                break
            elif np.all(np.absolute(W) < 1e-6):
                print('Found convergence at iterator %d on run %d'%(i,j))
                break
    return y, W
    #return np.dot(W, mixtures), W

def cichocki_Feedforward(mixtures, learningRate = 1e-2, runs = 5, decay = True, decayRate = 0.005):
    # FeedFoward
    I = np.identity(mixtures.shape[0])
    W = I
    y = np.zeros(mixtures.shape)
    dW = np.ones(W.shape) - I
    
    whitenedMixtures = whiten(mixtures)
    
    for j in np.arange(runs):
        for i in np.arange(mixtures.shape[1]):
            if decay:
                learning_rate = np.exp(-decayRate*(i+j))*learningRate
            else:
                learning_rate = learningRate
            input_ = np.reshape(whitenedMixtures[:,i], (mixtures.shape[0], 1))

            y[:,i] = np.reshape(np.dot(W, input_), (mixtures.shape[0],))
            gY = np.reshape(g(y[:,i]), (mixtures.shape[0],1))
            fY = np.reshape(f(y[:,i]), (mixtures.shape[0],1))

            dW = np.dot(I-np.dot(fY,np.transpose(gY)),W)
            W = W + learning_rate*dW
            if (np.isnan(W).any() == True):
                print('Lost convergence at iterator %d'%i)
                return y, W
            elif np.all(np.absolute(W) < 1e-6):
                print('Found convergence at iterator %d on run %d'%(i,j))
                return y, W
    return y, W
    #return np.dot(W, mixtures), W


def cichocki_Feedback(mixtures, learningRate = 1e-2, runs = 5, decay = True, decayRate  = 0.005):
    # Feedback
    I = np.identity(mixtures.shape[0])
    W = np.zeros((mixtures.shape[0], mixtures.shape[0]))
    y = np.zeros(mixtures.shape)
    dW = np.ones(W.shape) - I
   
    whitenedMixtures = whiten(mixtures)

    for j in np.arange(runs):
        for i in np.arange(mixtures.shape[1]):
            if decay:
                learning_rate = np.exp(-decayRate*(i+j))*learningRate
            else:
                learning_rate = learningRate
            inversa = inv(I+W)
            input_ = np.reshape(whitenedMixtures[:,i], (mixtures.shape[0], 1))

            y[:,i] = np.reshape(np.dot(inversa, input_), (mixtures.shape[0],))
            gY = np.reshape(g(y[:,i]), (mixtures.shape[0],1))
            fY = np.reshape(f(y[:,i]), (mixtures.shape[0],1))

            dW = np.dot((I+W),I-np.dot(fY,np.transpose(gY)))
            W = W - learning_rate*dW

            if (np.isnan(W).any() == True):
                print('Lost convergence at iterator %d'%i)
                return y, inv(I+W)
            elif np.all(np.absolute(W) < 1e-12):
                print('Found convergence at iterator %d on run %d'%(i,j))
                return y, inv(I+W)
    return y, inv(I+W)
    #return np.dot(W, mixtures), W