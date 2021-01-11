########################################
# Overview
#
# This file defines loss functions for 
# generalized linear models. These functions 
# are the negative log likelihoods using the 
# canonical links.
########################################
import numpy as np
from scipy.special import factorial, gamma

def SSE(Y, X, B):
    Y_Hat = np.matmul(X, B)
    Y_Hat.reshape([Y_Hat.shape[0],1])
    loss = np.sum( np.power(Y - Y_Hat, 2))
    
    return loss

########################################
# Mathematical background
# Step 1 Calculate eta
# Step 2 Inverse link
# step 3 Calculate negative log likelihood
########################################

def neg_ll_gaussian(Y, X, B):
    Y_Hat = np.matmul(X, B)
    Y_Hat.reshape([Y_Hat.shape[0],1])
    Y_Hat = Y_Hat # Inverse link
    
    n = X.shape[0]
    p = X.shape[1]
    sigma = np.sum(np.power(Y - Y_Hat, 2)) / (n-p)
    
    ll = -(n/2)*np.log(2*np.pi) - n*np.log(sigma) - (1/2*np.power(sigma, 2)) * np.sum(np.power(Y-Y_Hat,2))
    ll = -1 * ll
    
    return ll

def neg_ll_poisson(Y, X, B):
    Y_Hat = np.matmul(X, B)
    Y_Hat.reshape([Y_Hat.shape[0],1])
    Y_Hat = np.exp(Y_Hat) # Inverse link
    
    ll = Y * np.log(Y_Hat) - Y_Hat - np.log(factorial(Y))
    ll = np.sum(ll)
    ll = -1 * ll
    
    return ll

def neg_ll_bernoulli(Y, X, B):
    Y_Hat = np.matmul(X, B)
    Y_Hat.reshape([Y_Hat.shape[0],1])
    Y_Hat = np.exp(Y_Hat) / (1 + np.exp(Y_Hat)) # Inverse link
    
    ll = Y * np.log(Y_Hat) + (1 - Y) * np.log(1 - Y_Hat)
    ll = np.sum(ll)
    ll = -1 * ll
    
    return ll

def neg_ll_gamma(Y, X, B):
    Y_Hat = np.matmul(X, B)
    Y_Hat.reshape([Y_Hat.shape[0],1])
    Y_Hat = np.power(Y_Hat, -1) # Inverse link
    
    # Using method of moments estimate
    # MLE does not have closed form.
    # Don't want to write numerical method for MLE
    n = X.shape[0]
    p = X.shape[1]
    numerator = np.power(Y - Y_Hat, 2)
    denominator = np.power(Y_Hat, 2) * (n-p)
    phi = np.sum(numerator / denominator)
    
    ll = -1 * np.log(Y) - np.log(gamma(1/phi)) + (1/phi) * (np.log(Y) - np.log(Y_Hat) - np.log(phi)) - Y/(Y_Hat * phi)
    ll = np.sum(ll)
    ll = -1 * ll
    
    return ll