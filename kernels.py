#!/usr/bin/env python3

#Import libraries
from scipy.sparse import lil_matrix
from tqdm import tqdm
import numpy as np

class MismatchKernel:
    '''
    The mismatch Kernel
    Attributes :
        x_train_embeddings : Save training points embedding to avoid redundant computation during prediction
                             Format lil_matrix
        k : Length of kmers
        nb_mismatch : 0 - Spectrum Kernel
                      1 - Mismatches of 1 nucleotide are took in account
                      2 - Mismatches of 2 nucleotides are took in account
    '''
    def __init__(self, k, nb_mismatch):
        self.k = k
        self.x_train_embeddings = None
        self.nb_mismatch = nb_mismatch

    def kmers_one_off(self, kmer):
        """
        Find all kmers close to the input kmer up to one mismatch
        For example, 
            input : 'AC',
            excepted ouput : ['TC', 'CC', 'CG', 'AA', 'AT', 'AG']
        """
        kmers1 = []
        nucleotides = ['A','T','C','G']
        for index, caracter in enumerate(kmer):
            for nucleotide in nucleotides:
                if  nucleotide != caracter:
                    kmer1 = str(kmer[:index]) + nucleotide + str(kmer[index+1:])
                    kmers1.append(kmer1)
        return kmers1

    def kmers_two_off(self, kmer):
        """
        Find all kmers close to the input kmer up to two mismatches
        """
        #A set is used to skip duplicates
        kmers2 = set()
        kmers1 = self.kmers_one_off(kmer)
        for kmer1 in kmers1:
            for candidate in self.kmers_one_off(kmer1):
                if candidate != kmer and candidate not in kmers1:
                    kmers2.add(candidate)
        return kmers2

    def embed(self, x):
        '''
        x is a numpy array

        The function returns phi(x), a matrix containing embeddings of each sequence in x
        phi(x_i) has the same number of dimension than the number of possible kmer of lengths k
        phi(x_i) contains scores related to the appearance of each kmer in x_i
        '''
        #Embeddings are handled using a sparse data structure "lil_matrix"
        #This format is convenient for constructing sparse matrices incrementally.
        embeddings = lil_matrix((len(x), 4**self.k), dtype=float)
        mapping = {'A':0,'C':1,'T':2,'G':3}
        
        for idx in tqdm(range(len(x)), desc="Kernel Matrix Creation"):
            for i in range(len(x[idx][0]) - self.k + 1):
                seq = x[idx][0][i:i + self.k]   
                code = 0
                for power, caracter in enumerate(seq):
                    code += 4**(self.k - 1 - power)*mapping[caracter]
                embeddings[idx, code] += 10
                if self.nb_mismatch >= 1:
                    for seq1 in self.kmers_one_off(seq):
                        code = 0
                        for power, caracter in enumerate(seq1):
                            code += 4**(self.k - 1 - power)*mapping[caracter]
                        embeddings[idx, code] += 5
                if self.nb_mismatch >= 2:
                    for seq2 in self.kmers_two_off(seq):
                        code = 0
                        for power, caracter in enumerate(seq2):
                            code += 4**(self.k - 1 - power)*mapping[caracter]
                        embeddings[idx, code] += 1
        return embeddings

    def create_kernel_matrix(self, x):
        '''
        x is a numpy array corresponding to x_test during prediction and x_train during training. 
        The 2 use cases are separated for runtime efficiency. 
        Once the model is trained, x_train embeddings don't need to be computed again.
        '''
        if self.x_train_embeddings is None:
            self.x_train_embeddings = self.embed(x)
            #Lil_matrices are finally converted to csr format. 
            #It is another sparse data structure which is convenient for arithmetic operations 
            return np.array(np.dot(self.x_train_embeddings.tocsr(), self.x_train_embeddings.tocsr().T).todense()) 
        else:
            x_test_embeddings = self.embed(x)
            return np.array(np.dot(x_test_embeddings.tocsr(), self.x_train_embeddings.tocsr().T).todense())

    def create_partial_kernel_matrix(self, x_test, support_indices):
        '''
        x_test is a numpy array
        support_indices is a column numpy array of booleans, with values :
            True if the index corresponds to a support vector
            False otherwise
        '''
        x_test_embeddings = self.embed(x_test)
        partial_x_train_embeddings = self.x_train_embeddings[np.where(support_indices)[0], :]
        return np.array(np.dot(x_test_embeddings.tocsr(), partial_x_train_embeddings.tocsr().T).todense())

class PolynomialKernel:
    '''
    Polynomial Kernel
    Attributes
        degree : Degree of the polynomial kernel K(x, y) = <x, y>**degree
        x_train : Save training points 

    '''
    def __init__(self, degree):
        self.degree = degree
        self.x_train = None

    def create_kernel_matrix(self, x):
        '''
        x is a numpy array corresponding to x_test during prediction and x_train during training.
        Return a similarity matrix, measuring a similarity between vectors computed using the Polynomial Kernel.
        '''
        if self.x_train is None:
            self.x_train = x
            m = np.zeros((x.shape[0], x.shape[0]))
            for i, row1 in enumerate(x):
                for j, row2 in enumerate(x):
                    m[i, j] = np.dot(row1, row2)**self.degree
            return m   
        else:
            m = np.zeros((x.shape[0], self.x_train.shape[0]))
            for i, row1 in enumerate(x):
                for j, row2 in enumerate(self.x_train):
                    m[i, j] = np.dot(row1, row2)**self.degree
            return m    

class GaussianKernel:
    '''
    Gaussian Kernel
    Attributes
        gamma : Dispersion/scale hyparameter 
                It can be set equals to 1/(x_train.shape[1] * x_train.var()), 
                with x_train.shape[1] the dimension of data points and x_train.var() the points' variance/dispersion
        x_train : Save training points 
    '''
    def __init__(self, gamma):
        self.gamma = gamma
        self.x_train = None

    def create_kernel_matrix(self, x):
        '''
        x is a numpy array corresponding to x_test during prediction and x_train during training.
        Return a similarity matrix, measuring a similarity between vectors computed using the Gaussian Kernel.
        '''
        if self.x_train is None:
            self.x_train = x
            if self.gamma is None:
                self.gamma = 1/(self.x_train.shape[1] * self.x_train.var())
            m = np.zeros((x.shape[0], x.shape[0]))
            for i, row1 in enumerate(x):
                for j, row2 in enumerate(x):
                    m[i, j] = np.exp(-self.gamma * np.dot(row1 - row2, row1 - row2))
            return m   
        else:
            m = np.zeros((x.shape[0], self.x_train.shape[0]))
            for i, row1 in enumerate(x):
                for j, row2 in enumerate(self.x_train):
                    m[i, j] = np.exp(-self.gamma * np.dot(row1 - row2, row1 - row2))
            return m   


