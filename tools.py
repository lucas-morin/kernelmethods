#!/usr/bin/env python3

#Import libraries' functions
from Bio.Seq import Seq
import pandas as pd

def reverse_complement(seq):
    return seq.reverse_complement()

def augment(x_train, y_train):
    x_train_r = x_train.applymap(lambda x : str(reverse_complement(Seq(x))))
    x_train_r.index = range(len(x_train), 2*len(x_train))
    x_train = pd.concat([x_train, x_train_r])
    y_train_r = y_train.copy(deep=True)
    y_train_r.index = range(len(y_train), 2*len(y_train)) 
    y_train = pd.concat([y_train, y_train_r])
    return x_train, y_train

def read_embedded_data(file_name):
    x_train = pd.DataFrame(columns = ["Cluster " + str(id) for id in range(100)])
    with open(file_name) as f:
        lines = f.readlines()
        idx = 0
        for l in lines:
            x_train.loc[idx] = pd.Series(l.strip().split(" "), index=["Cluster " + str(id) for id in range(100)]).astype(float)
            idx = idx + 1
    return x_train