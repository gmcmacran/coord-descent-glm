########################################
# Overview
#
# This file defines a helper function that
# creates ideal data for a generalized 
# linear model
########################################
import pytest
import numpy as np

# Helper function to create dataset with
# true relationship between Y and X follows
# the glm model exactly.
#
# dist - string. One of "gaussian", "poisson", "bernoulli", "gamma"
# @pytest.fixture
def create_glm_dataset(dist):
    
    np.random.seed(0)
    
    X = np.random.rand(5000, 1) * 3
    X = np.hstack([X, np.ones([X.shape[0], 1])])
    
    B = np.ndarray([2, 1])

    
    if dist == "gaussian":
        B[0, 0] = 2
        B[1, 0] = 3
        
        mu = np.matmul(X, B)
        sigma = 1
        Y = np.random.normal(mu, sigma)
        
    elif dist == "poisson":
        B[0, 0] = .5
        B[1, 0] = 1
        
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
        Y = np.random.gamma(mu /.05, .05)
        
    X = np.delete(X, 1, axis = 1)
        
    return X, Y