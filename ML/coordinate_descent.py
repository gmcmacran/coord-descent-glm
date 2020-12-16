########################################
# Overview
#
# This file defines optimizer functions 
########################################
import numpy as np
import glm_negative_log_likelihoods as losses

# X predictors. numpy array
# Y response. numpy array
# loss loss function. See glm_negative_log_likelihoods
# shuffle bool. Should coordinates be shuffled every iteration? True is best practice
# step float. Step size
def coordinate_descent(X, Y, loss, shuffle, step):
    
    # Add intercept
    X = np.hstack([X, np.ones([X.shape[0], 1])])
    
    # set starting values
    B = np.ones([X.shape[1], 1], dtype = 'float64')
    # B[0, 0] = 2
    # B[1, 0] = 3
    
    
    # optimize
    for i in range(0, 1000):
        
        if shuffle == False:
            coords = np.arange(0, X.shape[1])
        else:
            coords = np.random.permutation(X.shape[1])
        
        # coords = range(0, 1)
        B_top_loop = B.copy()
        # coord = coords[0]
        # coord = coords[1]
        for coord in coords:
            # Step in both directions in current coordinate
            B_decrease = B.copy()
            B_decrease[coord, 0] = B_decrease[coord, 0] - step
            
            B_increase = B.copy()
            B_increase[coord, 0] = B_increase[coord, 0] + step
            
            loss_current = loss(Y, X, B)
            loss_decrease = loss(Y, X, B_decrease)
            loss_increase = loss(Y, X, B_increase)
            
            # Update B_current
            if loss_decrease < loss_current:
                B = B_decrease.copy()
            elif loss_increase < loss_current:
                B = B_increase.copy()
        #print(B)
        #print(loss(Y, X, B))
        
        # Update step if needed
        if (loss(Y, X, B_top_loop) == loss(Y, X, B)):
            step = .9 * step
    
    return B

# Curry optimizer
def make_glm_model(loss, shuffle, step):
    def model(X, Y):
        return(coordinate_descent(X, Y, loss, shuffle, step))
    return model

glm_gaussian = make_glm_model(loss = losses.neg_ll_gaussian, shuffle = True, step = .1)
glm_poisson = make_glm_model(loss = losses.neg_ll_poisson, shuffle = True, step = .1)
glm_bernoulli = make_glm_model(loss = losses.neg_ll_bernoulli, shuffle = True, step = .1)
glm_gamma = make_glm_model(loss = losses.neg_ll_gamma, shuffle = True, step = .1)

glm_SSE = make_glm_model(loss = losses.SSE, shuffle = True, step = .1)