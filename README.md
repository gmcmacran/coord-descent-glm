# Overview

## What is this library?
An implementation of the [generlized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model)

The models implemented are:
* Gaussian model with identity link.
* Binomial model with logit link.
* Poisson model with natural log link.
* Gamma model with inverse link.

## How are the models implemented?
Any supervised learning model can be defined in two pieces:
* An optimization function
* A loss function

This library uses [coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent) as the optimizer and 
[negative log likelihoods](https://en.wikipedia.org/wiki/Likelihood_function) as the loss functions.

## What does this library depend on?
* `numpy`: for matrices
* `scipy`: for a few mathematical operations

