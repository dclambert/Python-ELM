Python-ELM
==========

Extreme Learning Machine implementation in Python
Version  0.2

This is an implementation of the Extreme Learning Machine in python,
based on the scikit-learn machine learning library.

Distance and dot product based hidden layers are provided via the
RBFRandomLayer and SimpleRandomLayer classes respectively.

The SimpleRandomLayer provides the following activation functions:

    tanh, sine, tribas, sigmoid, hardlim

The RBFRandomLayer provides the following activation functions:

    gaussian, multiquadric and polyharmonic spline ('poly_spline')

In addition, each random hidden layer class can take a callable user
provided transfer function.  See the docstrings and the example ipython
notebook for details.

There's a little demo in plot_elm_comparison.py (based on scikit-learn's
plot_classifier_comparison).

Requires that scikit-learn be installed, along with its usual prerequisites,
and ipython to use elm_notebook.py (though it can be tweaked to run without
it).

This is a work in progress, it may be restructured as time goes	by.

- David C Lambert
  March, 2013
  [dcl -at- panix -dot- com]
