# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:51:29 2021

@author: Alena Edora
"""
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
    
from sklearn.metrics import accuracy_score, \
 confusion_matrix, ConfusionMatrixDisplay


df = pd.read_excel('classification_features.xlsx')

# Get names of indexes to drop
indexNames = df[df['Class'] == 'Banana'].index

# Delete these row indexes from dataFrame
df.drop(indexNames, inplace=True)

# Replacing Mango and Orange with 1 and -1
df = df.replace(['Mango','Orange'], [1,0])

# splitting the test and training data set
mango_df_train = df[df['Class']==1].head(18)
mango_df_test = df[df['Class']==1].tail(17)

orange_df_train = df[df['Class']==0].head(18) 
orange_df_test = df[df['Class']==0].tail(17)



df_train = pd.concat([mango_df_train, orange_df_train], axis=0)
df_test = pd.concat([mango_df_test, orange_df_test], axis=0)

train_samples = df_train.shape[0]   # no. of samples of train set
test_samples = df_test.shape[0]   # no. of samples of test set

# input features + bias
x1_train = df_train[['Normalized Hue','NormRound']].values
x0_train = np.ones((train_samples,1))
x_train = np.concatenate((x0_train,x1_train), axis=1)

x1_test = df_test[['Normalized Hue','NormRound']].values
x0_test = np.ones((test_samples,1))
x_test = np.concatenate((x0_test,x1_test), axis=1)

# true values
t_train = df_train[['Class']].values
t_test = df_test[['Class']].values

# defining the functions to use
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

def step(z):
    """
    only for accuracy purposes since the output values of the neural network are
    not exactly at 1s and 0s
    """
    y = []
    for i in z.T:
        if i > 0.5:
            y.append(1)
        else:
            y.append(0)
            
    return y

# initialization
eta = 0.01  # learning rate
epoch = 1000  # no. of times when all patterns have passed thru the network
hidden_node = 5    # no. of hidden nodes
output_node= 1   # no. of output nodes
feature_count = 3  # including bias

# plotting the cost function
SSE_cost = []


# weights init
w1 = np.random.rand(hidden_node, feature_count)  - 0.5  # (5,3)
w2 = np.random.rand(output_node, hidden_node) - 0.5 # (1,5) 

for i in range(epoch):
    # first layer
    a1 = np.dot(w1,x_train.T)                           
    z1 = np.array(relu(a1))

    # second layer
    a2 = np.dot(w2,z1)                           
    z2 = np.array(relu(a2))   # also yk

    # computing error of output unit
    delta_2 = drelu(a2) * (z2-t_train.T)  
    delta_1 = drelu(z1) * np.dot(delta_2.T,w2).T

    # computing the error derivatives of the samples
    dE_2 = np.dot(delta_2,z1.T)
    dE_1 = np.dot(delta_1,x_train)

    # computing for weight change
    w2 += -eta * dE_2
    w1 += -eta * dE_1
    
    err = 0.5 * np.sum((z2-t_train.T)**2)
    SSE_cost.append(err)
   
# testing the accuracy of the model

# y_pred = step(z2)
aj = np.dot(w1,x_test.T)                           
zj = np.array(relu(aj))

# second layer
ak = np.dot(w2,zj)                           
zk = np.array(relu(ak))   # also yk

y_pred = step(zk)

# prints accuracy of the model
print('Accuracy: ', accuracy_score(y_pred,t_test))

cm = confusion_matrix(y_pred, t_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix-acc-0.8.png')
plt.show()

plt.plot(np.arange(0,epoch,1), SSE_cost)
plt.xlabel('No. of Epochs')
plt.ylabel('Cost Function')
plt.title('2-layer Neural Network Cost Function for Fruit Classification')
plt.savefig('cost_func_fruits.png')
plt.show()

