#!/usr/bin/env python3

#Import libraries' functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

#Import source code functions
from tools import *
from models import *
from kernels import *

#Define model hyperparameters best suited (To modify regarding the model used)
lambda_ = 100 #SVR
#lambda_ = 1e-10 #KLR
#k = 9 #Mismatch Kernel 
#nb_mismatch = 0 #Mismatch Kernel 
degree = 7 #Polynomial Kernel 

#Choose a dataset
i = 1

#Read training data
#x_train = pd.read_csv(f"./data/Xtr{i}.csv", index_col=0)
#y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0)
x_train = read_embedded_data(f"./data/Xtr{i}_mat100.csv") 
y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0)

#Augment training data (optional)
#x_train, y_train = augment(x_train, y_train)

#Split training data in training and validation sets
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)

#Create kernel
#kernel = MismatchKernel(k = k, nb_mismatch = nb_mismatch)
#kernel = GaussianKernel()
kernel = PolynomialKernel(degree = degree)

#Create model
model = SVR(lambda_ = lambda_, kernel = kernel) 
#model = KLR(lambda_=lambda_, kernel=kernel) 

#Change labels from 0, 1 to -1, 1 
y_train = y_train.applymap(lambda x: 1 if (x == 1) else -1)

#Train the model
model.fit(x_train, y_train)

#Predict new labels
predictions_train = (model.predict(x_train) > 0).astype(int)  
predictions_validation = (model.predict(x_validation) > 0).astype(int)  

#Come back to 0, 1 labels
y_train = y_train.applymap(lambda x: 1 if (x == 1) else 0)

#Measure performances
print(f"The training accuracy is {accuracy_score(y_train, predictions_train)}")
print(f"The validation accuracy is {accuracy_score(y_validation, predictions_validation)}")