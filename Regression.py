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

##Load wine testing UCI data data
data = np.genfromtxt('winequality-white.csv', dtype = float, delimiter = ';')

#Delete heading
data = np.delete(data,0,0)

x = data[:,:11]
y = data[:,-1]

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=42)

#===============================================================

def calculateE(y,t):

    #Calculate RMSE
    return mean_squared_error(y, t)


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
    w = random.rand(11,L)

    #Initialize model
    model = hpelm.ELM(11,1)
    model.add_neurons(L,'sigm')
    
    start_time = time.time()
    
    #Train model
    model.train(x_train,y_train,'r')
    
    Train_T.append(time.time() - start_time)
    
    #Calculate output weights and intermediate layer
    BL_HL = model.predict(x_test)
    
    #Calculate new EMSE
    E = calculateE(y_test,BL_HL)
    
    Test_E.append(E)
    
    #Print result
    print("Hidden Node",L,"RMSE :",E)
    
#===================================================================

#Find best RMSE
L = Test_E.index(min(Test_E)) + 1

print()
print()
print()
print()

#Define model 
model = hpelm.ELM(11,1)
model.add_neurons(L,'sigm')

start_time = time.time()
model.train(x_train,y_train,'r')
print('Training Time :',time.time() - start_time)

start_time = time.time()
BL_HL = model.predict(x_train) 
print('Testing Time :',time.time() - start_time)

#Calculate training RMSE  
E = calculateE(y_train,BL_HL)
print('Training RMSE :',E)
print('Testing RMSE  :',min(Test_E))

#===================================================================

#Plot Data

plt.subplot(1, 2, 1)    #Generate graph for ANN
plt.plot(range(1,Lmax+1),Test_E)
plt.title('Testing RMSE')
plt.xlabel('Number of Neurons in hidden layer')
plt.ylabel('Testing RMSE')


plt.subplot(1, 2, 2)    #Generate graph for CNN
plt.plot(range(1,Lmax+1),Train_T)
plt.title('Training Time')
plt.xlabel('Number of Neurons in hidden layer')
plt.ylabel('Training Time')


plt.show()
