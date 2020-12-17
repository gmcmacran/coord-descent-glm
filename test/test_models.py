import numpy as np
import create_glm_data as data
import ML.coordinate_descent as ai

def test_gaussian_model():
    X, Y = data.create_glm_dataset("gaussian")
    B = ai.glm_gaussian(X, Y)
    
    target = np.array([[2],
                      [3]])
    
    np.testing.assert_array_almost_equal(B, target, 1)

def test_poisson_model():
    X, Y = data.create_glm_dataset("poisson")
    B = ai.glm_poisson(X, Y)
    
    target = np.array([[.5],
                      [1]])
    
    np.testing.assert_array_almost_equal(B, target, 1)
    
def test_bernoulli_model():
    X, Y = data.create_glm_dataset("bernoulli")
    B = ai.glm_bernoulli(X, Y)
    
    target = np.array([[2],
                      [-.7]])
    
    np.testing.assert_array_almost_equal(B, target, 1)
    
def test_gamma_model():
    X, Y = data.create_glm_dataset("gamma")
    B = ai.glm_gamma(X, Y)
    
    target = np.array([[2],
                      [3]])
    
    np.testing.assert_array_almost_equal(B, target, 1)

test_gaussian_model()
test_poisson_model()
test_bernoulli_model()
test_gamma_model()