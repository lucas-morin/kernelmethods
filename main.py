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

Yte = []

#Define model hyperparameters best suited for each dataset 
#(To modify regarding the model used)
#SVR
#lambda_ = [100, 100, 100]
#k = [9, 9, 8]
#KLR
lambda_ = [1e-10, 1e-10, 1e-10]

#Iterate over the 3 data sets
for i in range(3):

    #Read training data
    #x_train = pd.read_csv(f"./data/Xtr{i}.csv", index_col=0)
    #y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0)
    x_train = read_embedded_data(f"./data/Xtr{i}_mat100.csv") 
    y_train = pd.read_csv(f"./data/Ytr{i}.csv", index_col=0)

    #Augment training data (optional)
    #x_train, y_train = augment(x_train, y_train)
    
    #Read test input
    #x_test = pd.read_csv(f"./data/Xte{i}.csv", index_col=0)
    x_test = read_embedded_data(f"./data/Xte{i}_mat100.csv") 

    #Create kernel
    #kernel = MK(k = k[i])
    kernel = GK()

    #Create model
    #model = SVR(lambda_ = lambda_[i], kernel = kernel) 
    model = KLR(lambda_=lambda_[i], kernel=kernel) 

    #Change labels from 0, 1 to -1, 1 
    #y_train = y_train.applymap(lambda x: 1 if (x == 1) else -1)

    #Train the model
    model.fit(x_train, y_train)

    #Predict new labels
    Yte = np.concatenate((Yte, (model.predict(x_test) > 0).astype(int)))
    
Yte = pd.DataFrame(Yte, index=range(3000)).astype(int)
Yte.to_csv(path_or_buf="./data/Yte.csv", header=['Bound'], index_label="Id", index=True)






