Python-ELM
==========

Extreme Learning Machine implementation in Python
Version  0.3

This is an implementation of the [Extreme Learning Machine](http://www.extreme-learning-machines.org) [1][2] in Python, based on [scikit-learn](http://scikit-learn.org).

It's a work in progress, so things can/might/will change.

__David C. Lambert__  
__dcl [at] panix [dot] com__  

__Copyright © 2013__  
__License: Simple BSD__

Files
-----
####__random_layer.py__

Contains the __RandomLayer__, __MLPRandomLayer__, __RBFRandomLayer__ and __GRBFRandomLayer__ classes.

RandomLayer is a transformer that creates a feature mapping of the
inputs that corresponds to a layer of hidden units with randomly 
generated components.

The transformed values are a specified function of input activations
that are a weighted combination of dot product (multilayer perceptron)
and distance (rbf) activations:

	  input_activation = alpha * mlp_activation + (1-alpha) * rbf_activation

	  mlp_activation(x) = dot(x, weights) + bias
	  rbf_activation(x) = rbf_width * ||x - center||/radius

_mlp_activation_ is multi-layer perceptron input activation  

_rbf_activation_ is radial basis function input activation

_alpha_ and _rbf_width_ are specified by the user

_weights_ and _biases_ are taken from normal distribution of
mean 0 and sd of 1

_centers_ are taken uniformly from the bounding hyperrectangle
of the inputs, and

	radius = max(||x-c||)/sqrt(n_centers*2)

(All random components can be supplied by the user by providing entries in the dictionary given as the _user_components_ parameter.)

The input activation is transformed by a transfer function that defaults
to numpy.tanh if not specified, but can be any callable that returns an
array of the same shape as its argument (the input activation array, of
shape [n_samples, n_hidden]).

Transfer functions provided are:

*	sine
*	tanh
*	tribas
*	inv_tribas
*	sigmoid
*	hardlim
*	softlim
*	gaussian
*	multiquadric
*	inv_multiquadric

MLPRandomLayer and RBFRandomLayer classes are just wrappers around the RandomLayer class, with the _alpha_ mixing parameter set to 1.0 and 0.0 respectively (for 100% MLP input activation, or 100% RBF input activation)
	
The RandomLayer, MLPRandomLayer, RBFRandomLayer classes can take a callable user
provided transfer function.  See the docstrings and the example ipython
notebook for details.

The GRBFRandomLayer implements the Generalized Radial Basis Function from [[3]](http://sci2s.ugr.es/keel/pdf/keel/articulo/2011-Neurocomputing1.pdf)

####__elm.py__

Contains the __ELMRegressor__, __ELMClassifier__, __GenELMRegressor__, and __GenELMClassifier__ classes.

GenELMRegressor and GenELMClassifier both take *RandomLayer instances as part of their contructors, and an optional regressor (conforming to the sklearn API)for performing the fit (instead of the default linear fit using the pseudo inverse from scipy.pinv2).
GenELMClassifier is little more than a wrapper around GenELMRegressor that binarizes the target array before performing a regression, then unbinarizes the prediction of the regressor to make its own predictions.

The ELMRegressor class is a wrapper around GenELMRegressor that uses a RandomLayer instance by default and exposes the RandomLayer parameters in the constructor.  ELMClassifier is similar for classification.

####__plot_elm_comparison.py__

A small demo ()based on scikit-learn's plot_classifier_comparison) that shows the decision functions of a couple of different instantiations of the GenELMClassifier on three different datasets.

####__elm_notebook.py__

An IPython notebook, illustrating several ways to use the __\*ELM*__ and __\*RandomLayer__ classes.

Requirements
------------

Written using Python 2.7.3, numpy 1.6.1, scipy 0.10.1, scikit-learn 0.13.1 and ipython 0.12.1

References
----------
```
[1] http://www.extreme-learning-machines.org

[2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
          2006.
          
[3] Fernandez-Navarro, et al, "MELM-GRBF: a modified version of the  
          extreme learning machine for generalized radial basis function  
          neural networks", Neurocomputing 74 (2011), 2502-2510
```

