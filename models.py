#!/usr/bin/env python3

#Import libraries' functions
from cvxopt import solvers
from cvxopt import matrix
from numpy import linalg as LA
import numpy as np

class KLR:
    '''
    Kernel Logistic Regression
    Attributes:
        alpha : Solution of the optimization problem
        lambda_ : Regularization parameter
        kernel : Used kernel 
        fitted : Indicator describing whether the model is already fitted 
        x_fit : Save training points 
    '''
    def __init__(self, lambda_, kernel):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = None
        self.x_fit = None
        self.fitted = False
    
    def fit(self, x_train, y_train):
        '''
        x_train and y_train are dataframes
        '''
        #Change data type
        x_train = x_train.values
        y_train = y_train['Bound'].values

        n, m = x_train.shape

        #We compute the closed form solution of the optimization problem
        self.kernel.gamma = 1/(m * x_train.var())
        K = self.kernel.create_kernel_matrix(x_train, x_train)
        self.alpha = np.dot(LA.inv((K + self.lambda_ * n * np.identity(n))), y_train)
        self.fitted = True
        self.x_fit = x_train
        return self

    def predict(self, x_test):
        '''
        x_test is a dataframe
        '''
        #Change data type
        x_test = x_test.values

        if not(self.fitted):
            raise Error("The estimator needs to be fitted before calling self.predict().")

        K = self.kernel.create_kernel_matrix(x_test, self.x_fit)
        return np.dot(K, self.alpha)
    
    
class KRR:
    """
    Kernel Ridge Regression
    """
    #TO DO

class SVR:
    '''
    Support Vector Regression

    Attributes :
        alpha : Solution of the optimization problem
        lambda_ : Regularization parameter
        loss : Used loss function
        kernel : Used kernel 
        fitted : Indicator describing whether the model is already fitted 
    '''

    def __init__(self, lambda_, kernel):
        self.lambda_ = lambda_
        self.kernel = kernel
        self.loss = 'squared_hinge'
        self.alpha = None
        self.fitted = False
        self.support_indices = None

    def fit(self, x_train, y_train):
        '''
        x_train is a dataframe
        y_train is a dataframe

        Solve the optimization problem.
        Find the separation boundary between the 2 classes, for data points in the training set.
        '''
        #Change input types
        x_train = x_train.values 
        y_train = y_train["Bound"].values.astype(np.double)

        #Get Kernel Matrix
        K = self.kernel.create_kernel_matrix(x_train)

        n, m = x_train.shape
        diag_y = np.diag(y_train)
        idt = np.identity(n)
        
        if self.loss == 'hinge':
            P = 1 / (2*self.lambda_) * np.dot(diag_y, np.dot(K, diag_y))
            q = - np.ones(n)
            G = np.concatenate((idt, -idt))
            h = np.concatenate((np.ones(n) / n, np.zeros(n)))

            #The solution of the dual problem is called mu. (lagrange multipliers)
            #The optimization problem is solved using an iterative method : "cvxopt" quadratic program solver.
            mu = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))['x']
            self.alpha = np.dot(diag_y, mu) / (2*self.lambda_)
            self.support_indices = abs(self.alpha) < 1e-5
            
            
        elif self.loss == 'squared_hinge':
            P = (K + self.lambda_ * n * idt)
            q = -y_train
            G = -diag_y
            h = np.zeros(n)

            #Here, we directly solve the primal problem
            self.alpha = np.array(solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))['x'])
            self.support_indices = abs(self.alpha) < 5.5e-6
            
        self.fitted = True
        return self

    def predict(self, x_test):
        '''
        x_test is a dataframe

        For fast classification, use "fast_predict" instead. It only involves support vectors.
        
        An alternative for linear kernels would be to write the prediction function as f(x) = dot_product(x, w) + b,    
        w and b being the parameters of the separating hyperplane.
        '''
        if not self.fitted:
            raise SVMNotFitError 
        
        #Change data type
        x_test = x_test.values
        
        K = self.kernel.create_kernel_matrix(x_test)
    
        #The prediction function is expressed using the representer theorem
        return np.dot(K, self.alpha).reshape(-1)

    def fast_predict(self, x_test):
        '''
        x_test is a dataframe
        '''
        if not self.fitted:
            raise SVMNotFitError 
        
        #Change data type
        x_test = x_test.values

        partial_alpha = self.alpha[self.support_indices]
        partial_alpha = partial_alpha[:, np.newaxis]
        partial_K = self.kernel.create_partial_kernel_matrix(x_test, self.support_indices)
        return np.dot(partial_K, partial_alpha).reshape(-1)

class SVMNotFitError(Exception):
    """The estimator needs to be fitted before calling self.predict()."""