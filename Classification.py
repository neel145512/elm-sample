#Import libraries

import hpelm
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from numpy import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

#Lists to store results
Train_T = []
Test_E  = []

#Load mnist data
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#Scale data
x_train = x_train.astype(np.float32) / 255.0
x_train = x_train.reshape(-1,28*28)

x_test = x_test.astype(np.float32) / 255.0
x_test = x_test.reshape(-1,28*28)

# 1 hot encoding
y_train = to_categorical(y_train,10).astype(np.float32)
y_test = to_categorical(y_test,10).astype(np.float32)

#===============================================================

def calculateE(y,t):

    p = np.zeros_like(t)
    p[np.arange(len(t)), t.argmax(1)] = 1

    hit = 0
    miss = 0

    #Calculate accuracy
    for i in range(len(t)):

        if np.where(p[i]==1)==np.where(y[i]==1):
            hit = hit + 1
        else:
            miss = miss + 1
    
    return hit/(hit+miss)
    

#===============================================================
#Initialization

Lmax = 40
L = 0
E = 0
ExpectedAccuracy = 0

while L < Lmax and E >= ExpectedAccuracy:

    #Increase Node
    L = L + 1
    
    #Calculate Random weights, they are  already addded into model using hpelm library
    w = random.rand(784,L)

    #Initialize model
    model = hpelm.ELM(28*28,10)
    model.add_neurons(L,'sigm')
    
    start_time = time.time()
    
    #Train model
    model.train(x_train,y_train,'ml')
    
    Train_T.append(time.time() - start_time)
    
    #Calculate output weights and intermediate layer
    BL_HL = model.predict(x_test)
    
    #Calculate new accuracy
    E = calculateE(y_test,BL_HL)
    
    Test_E.append(E)
    
    #Print result
    print("Hidden Node",L,"Accuracy :",E)

#===================================================================

#Find best accuracy
L = Test_E.index(max(Test_E)) + 1

#Define model 
model = hpelm.ELM(28*28,10)
model.add_neurons(L,'sigm')

start_time = time.time()
model.train(x_train,y_train,'ml')
print('Training Time :',time.time() - start_time)

start_time = time.time()
BL_HL = model.predict(x_train) 
print('Testing Time :',time.time() - start_time)

#Calculate training accuracy 
E = calculateE(y_train,BL_HL)
print('Training Accuracy :',E)
print('Testing Accuracy  :',max(Test_E))

#===================================================================

#Plot Data

plt.subplot(1, 2, 1)    #Generate graph for ANN
plt.plot(range(1,Lmax+1),Test_E)
plt.title('Testing Accuracy')
plt.xlabel('Number of Neurons in hidden layer')
plt.ylabel('Testing Accuracy')


plt.subplot(1, 2, 2)    #Generate graph for CNN
plt.plot(range(1,Lmax+1),Train_T)
plt.title('Training Time')
plt.xlabel('Number of Neurons in hidden layer')
plt.ylabel('Training Time')


plt.show()
