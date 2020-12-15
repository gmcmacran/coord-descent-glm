########################################
# Overview
#
# This file defines optimizer functions 
########################################
import numpy as np
import glm_negative_log_likelihoods as losses

def coordinate_descent(X, Y, B, shuffle, loss, step):
    
    # Add intercept
    X = np.hstack([X, np.ones([X.shape[0], 1])])
    
    # optimize
    for i in range(0, 100):
        
        if shuffle == False:
            coords = np.arange(0, X.shape[1])
        else:
            coords = np.random.permutation(X.shape[1])
        
        B_top_loop = B.copy()
        for coord in coords:
            # Step in both directions in current coordinate
            B_decrease = B.copy()
            B_decrease[coord] = B_decrease[coord] - step
            
            B_increase = B.copy()
            B_increase[coord] = B_increase[coord] + step
            
            loss_current = loss(Y, X, B)
            loss_decrease = loss(Y, X, B_decrease)
            loss_increase = loss(Y, X, B_increase)
            
            # Update B_current
            if loss_decrease < loss_current:
                B = B_decrease.copy()
            elif loss_increase < loss_current:
                B = B_increase.copy()
        
        # Update step if needed
        if (loss(Y, X, B_top_loop) == loss(Y, X, B)):
            step = .9 * step
    
    return B

def make_glm_model(loss, shuffle, step):
    def model(X, Y, B):
        return(coordinate_descent(X, Y, B, shuffle, loss, step))
    return model

glm_gaussian = make_glm_model(loss = losses.neg_ll_gaussian, shuffle = True, step = .1)
glm_poisson = make_glm_model(loss = losses.neg_ll_poisson, shuffle = True, step = .1)
glm_bernoulli = make_glm_model(loss = losses.neg_ll_bernoulli, shuffle = True, step = .1)
glm_gamma = make_glm_model(loss = losses.neg_ll_gamma, shuffle = True, step = .1)