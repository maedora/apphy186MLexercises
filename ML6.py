# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:31:10 2021

@author: Alena Edora
"""
import numpy as np
import matplotlib.pyplot as plt

# define thex and y input

x1 = np.linspace(0,1,100)   # input vector
x1 = x1.reshape((x1.shape[0],1))     

x0 = np.ones(x1.shape)        # bias
x0 = x0.reshape((x0.shape[0],1))

# joining x1 and x2 features
x = np.concatenate((x0,x1), axis=1)

#t = 0.5*np.sin(2*np.pi*x1) + 0.5  # true value

t = 0.5*np.tanh(2*np.pi*x1)

# defining activation functions
def tanh(z):
    g = np.tanh(z)
    return g

def dtanh(g):
    dg = (1 - g**2)
    return dg

def sigmoid(z):
    g = (1/(1+np.exp(-z))) 
    return g 


def dsigmoid(g):
    dg = g * (1 - g)
    return dg

def linear(z):
    g = z
    return g

def dlinear(z):
    dg = np.ones(z.shape)
    return dg

def relu(z):
    g = np.maximum(0,z)
    return g

def drelu(z):
    dg = np.maximum(0,1)
    return dg

# initialization
eta = 0.05  # learning rate
epoch = 10000  # no. of times when all patterns have passed thru the network
hidden_node = 5    # no. of hidden nodes
output_node= 1   # no. of output nodes
feature_count = 2

# plotting the cost function
SSE_cost = []


# weights init

w1 = np.random.rand(hidden_node, feature_count) - 1   # (2,2)
w2 = np.random.rand(output_node, hidden_node) - 1 # (1,3) 

for i in range(epoch):
    # first layer
    a1 = np.dot(w1,x.T)                           
    z1 = np.array(sigmoid(a1))

    # second layer
    a2 = np.dot(w2,z1)                           
    z2 = np.array(linear(a2))   # also yk

    # computing error of output unit
    delta_2 = dlinear(a2) * (z2-t.T)  
    delta_1 = dsigmoid(z1) * np.dot(delta_2.T,w2).T

    # computing the error derivatives of the samples
    dE_2 = np.dot(delta_2,z1.T)
    dE_1 = np.dot(delta_1,x)

    # computing for weight change
    w2 += -eta * dE_2
    w1 += -eta * dE_1
    
    err =0.5 * np.sum((z2.T-t)**2)
    SSE_cost.append(err)
    

plt.plot(x1,z2.T, label='Predicted Value')
plt.plot(x1,t, label='True Value')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.title('Predicting Tanh Function using a 2-layer Neural Network')
plt.savefig('sigmoid_100samples-0.5.png')
plt.show()

plt.plot(np.arange(0,epoch,1), SSE_cost)
plt.xlabel('No. of Iterations')
plt.ylabel('Cost Function')
plt.legend()
plt.title('2-layer Neural Network Cost Function for Learning Sine Function')
plt.savefig('cost_func.png')

plt.show()