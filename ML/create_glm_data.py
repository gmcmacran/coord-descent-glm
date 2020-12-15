########################################
# Overview
#
# This file defines a helper function that
# creates ideal data for a generalized 
# linear model
########################################
import numpy as np

# WIll be converting to pytest in future
# Right now, just manually testing here.
import glm_negative_log_likelihoods as losses
import coordinate_descent as cd

def create_glm_dataset(dist):
    
    X = np.random.rand(500, 1)
    X = np.hstack([X, np.ones([X.shape[0], 1])])
    
    B = np.ndarray([2, 1])

    
    if dist == "gaussian":
        B[0, 0] = 2
        B[1, 0] = 3
        
        mu = np.matmul(X, B)
        sigma = .5
        Y = np.random.normal(mu, sigma)
        
    elif dist == "poisson":
        B[0, 0] = 2
        B[1, 0] = 3
        
        mu = np.matmul(X, B)
        mu = np.exp(mu)
        Y = np.random.poisson(mu)
        
    elif dist == "bernoulli":
        B[0, 0] = 2
        B[1, 0] = -.7
        
        mu = np.matmul(X, B)
        mu = np.exp(mu) / (1 + np.exp(mu))
        Y = np.random.binomial(1, mu)
        
    elif dist == "gamma":
        B[0, 0] = 2
        B[1, 0] = 3
        
        mu = np.matmul(X, B)
        mu = 1 / mu
        Y = np.random.gamma(mu /.05, 1 /.05)
        
    X = np.delete(X, 1, axis = 1)
        
    return X, Y

X, Y = create_glm_dataset("gaussian")
B_start = np.array([0.1, 0.1])
B_end = cd.glm_gaussian(X, Y, B_start)