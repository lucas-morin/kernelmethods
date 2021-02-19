#!/usr/bin/env python3

#Import libraries' functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Import source code functions
from tools import *
from models import *
from kernels import *

#Yte stores submission predictions
Yte = []

#Define model hyperparameters best suited for each dataset (To comment/uncomment)
lambda_ = [100, 100, 100] #SVR
#lambda_ = [1e-10, 1e-10, 1e-10] #KLR
k = [9, 9, 8] #Mismatch Kernel 
nb_mismatch = 2 #Mismatch Kernel 
#degree = [4, 4, 4] #Polynomial Kernel 
#gamma = [None, None, None] #Gaussian Kernel 

#Iterate over the 3 data sets
for i in range(3):

    #Read training data (To comment/uncomment)
    x_train = pd.read_csv(f"./data/Xtr{i}.csv", index_col=0) #Raw sequences
    y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0) #Raw sequences
    #x_train = read_embedded_data(f"./data/Xtr{i}_mat100.csv") #Numerical data
    #y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0) #Numerical data

    #Augment training data (Only for raw sequences)
    x_train, y_train = augment(x_train, y_train)
    
    #Read test input (To comment/uncomment)
    x_test = pd.read_csv(f"./data/Xte{i}.csv", index_col=0) #Raw sequences
    #x_test = read_embedded_data(f"./data/Xte{i}_mat100.csv") #Numerical data

    #Create kernel (To comment/uncomment)
    kernel = MismatchKernel(k = k[i], nb_mismatch = nb_mismatch)
    #kernel = GaussianKernel(gamma = gamma[i])
    #kernel = PolynomialKernel(degree = degree[i])

    #Create model (To comment/uncomment)
    model = SVR(lambda_ = lambda_[i], kernel = kernel) 
    #model = KRR(lambda_=lambda_[i], kernel=kernel) 

    #Change labels from 0, 1 to -1, 1 (Only for SVM)
    y_train = y_train.applymap(lambda x: 1 if (x == 1) else -1) 

    #Train the model
    model.fit(x_train, y_train)

    #Predict new labels (To comment/uncomment)
    #Yte = np.concatenate((Yte, (model.predict(x_test) > 1/2).astype(int))) #KRR
    Yte = np.concatenate((Yte, (model.predict(x_test) > 0).astype(int))) #SVM

#Submission is stored in ./data/Yte.csv 
Yte = pd.DataFrame(Yte, index=range(3000)).astype(int)
Yte.to_csv(path_or_buf="./data/Yte.csv", header=['Bound'], index_label="Id", index=True)






