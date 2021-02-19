#!/usr/bin/env python3

#Import libraries' functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Import source code functions
from tools import *
from models import *
from kernels import *

#Define model hyperparameters best suited (To comment/uncomment)
lambda_ = 100 #SVR
#lambda_ = 1e-10 #KLR
k = 9 #Mismatch Kernel 
nb_mismatch = 1 #Mismatch Kernel 
#degree = 7 #Polynomial Kernel 
#gamma = None #Gaussian Kernel 

#Choose a dataset
i = 2

#Read anotated data (To comment/uncomment)
x_train = pd.read_csv(f"./data/Xtr{i}.csv", index_col=0) #Raw sequences
y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0) #Raw sequences
#x_train = read_embedded_data(f"./data/Xtr{i}_mat100.csv") #Numerical data
#y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0) #Numerical data

#Augment anotated data (Only for raw sequences)
x_train, y_train = augment(x_train, y_train)

#Split anotated data in training and validation sets
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)

#Create kernel (To comment/uncomment)
kernel = MismatchKernel(k = k, nb_mismatch = nb_mismatch)
#kernel = GaussianKernel(gamma = gamma)
#kernel = PolynomialKernel(degree = degree)

#Create model (To comment/uncomment)
model = SVR(lambda_ = lambda_, kernel = kernel) 
#model = KRR(lambda_=lambda_, kernel=kernel) 

#Change labels from 0, 1 to -1, 1 (Only for SVM)
y_train = y_train.applymap(lambda x: 1 if (x == 1) else -1)

#Train the model
model.fit(x_train, y_train)

#Predict new labels (To comment/uncomment)
predictions_train = (model.predict(x_train) > 0).astype(int) #SVM
predictions_validation = (model.predict(x_validation) > 0).astype(int) #SVM
#predictions_train = (model.predict(x_train) > 1/2).astype(int) #KRR
#predictions_validation = (model.predict(x_validation) > 1/2).astype(int) #KRR

#Come back to 0, 1 labels 
y_train = y_train.applymap(lambda x: 1 if (x == 1) else 0)

#Measure performances
print(f"The training accuracy is {accuracy_score(y_train, predictions_train)}")
print(f"The validation accuracy is {accuracy_score(y_validation, predictions_validation)}")