# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

# Demo python notebook for elm and random_hidden_layer modules
#
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD

# <codecell>

from time import time
from sklearn.cluster import k_means
from elm import ELMClassifier, ELMRegressor, SimpleELMClassifier, SimpleELMRegressor
from random_hidden_layer import SimpleRandomHiddenLayer, RBFRandomHiddenLayer

# <codecell>

def make_toy():
    x = np.arange(0.25,20,0.1)
    y = x*np.cos(x)+np.random.randn(x.shape[0])
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    return x, y

# <codecell>

def res_dist(x, y, e, n_runs=100, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    test_res = []
    train_res = []
    start_time = time()

    for i in xrange(n_runs):
        e.fit(x_train, y_train)
        train_res.append(e.score(x_train, y_train))
        test_res.append(e.score(x_test, y_test))
        if (i%5 == 0): print "%d"%i,

    print "\nTime: %.3f secs" % (time() - start_time)

    print "Test Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(test_res), mean(test_res), max(test_res), std(test_res))
    print "Train Min: %.3f Mean: %.3f Max: %.3f SD: %.3f" % (min(train_res), mean(train_res), max(train_res), std(train_res))
    print
    return (train_res, test_res)

# <codecell>

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression

stdsc = StandardScaler()

iris = load_iris()
irx, iry = stdsc.fit_transform(iris.data), iris.target
irx_train, irx_test, iry_train, iry_test = train_test_split(irx, iry, test_size=0.2)

digits = load_digits()
dgx, dgy = stdsc.fit_transform(digits.data), digits.target
dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.2)

diabetes = load_diabetes()
#dbx, dby = stdsc.fit_transform(diabetes.data), stdsc.fit_transform(diabetes.target)
dbx, dby = stdsc.fit_transform(diabetes.data), diabetes.target
dbx_train, dbx_test, dby_train, dby_test = train_test_split(dbx, dby, test_size=0.2)

mrx, mry = make_regression(n_samples=2000)
mrx_train, mrx_test, mry_train, mry_test = train_test_split(mrx, mry, test_size=0.2)

xtoy, ytoy = make_toy()
xtoy, ytoy = stdsc.fit_transform(xtoy), stdsc.fit_transform(ytoy)
xtoy_train, xtoy_test, ytoy_train, ytoy_test = train_test_split(xtoy, ytoy, test_size=0.2)
plot(xtoy, ytoy)

# <codecell>

# RBF tests
elmc = ELMClassifier(RBFRandomHiddenLayer(activation_func='gaussian'))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

elmc = ELMClassifier(RBFRandomHiddenLayer(activation_func='poly_spline', gamma=2))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

elmc = ELMClassifier(RBFRandomHiddenLayer(activation_func='multiquadric'))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

# Simple tests
elmc = ELMClassifier(SimpleRandomHiddenLayer(activation_func='sine'))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

elmc = ELMClassifier(SimpleRandomHiddenLayer(activation_func='tanh'))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

elmc = ELMClassifier(SimpleRandomHiddenLayer(activation_func='tribas'))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

elmc = ELMClassifier(SimpleRandomHiddenLayer(activation_func='sigmoid'))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

elmc = ELMClassifier(SimpleRandomHiddenLayer(activation_func='hardlim'))
tr,ts = res_dist(irx, iry, elmc, n_runs=100, random_state=0)

# <codecell>

hardlim = (lambda a: np.array(a > 0.0, dtype=float))
tribas = (lambda a: np.clip(1.0 - np.fabs(a), 0.0, 1.0))
elmr = ELMRegressor(SimpleRandomHiddenLayer(random_state=0, activation_func=tribas))
elmr.fit(xtoy_train, ytoy_train)
print elmr.score(xtoy_train, ytoy_train), elmr.score(xtoy_test, ytoy_test)
plot(xtoy, ytoy, xtoy, elmr.predict(xtoy))

# <codecell>

rhl = SimpleRandomHiddenLayer(n_hidden=200)
elmr = ELMRegressor(hidden_layer=rhl)
tr, ts = res_dist(mrx, mry, elmr, n_runs=20, random_state=0)

# <codecell>

rhl = RBFRandomHiddenLayer(n_hidden=15, gamma=0.25)
elmr = ELMRegressor(hidden_layer=rhl)
elmr.fit(xtoy_train, ytoy_train)
print elmr.score(xtoy_train, ytoy_train), elmr.score(xtoy_test, ytoy_test)
plot(xtoy, ytoy, xtoy, elmr.predict(xtoy))

# <codecell>

nh = 10
(ctrs, _, _) = k_means(xtoy_train, nh)
unit_rs = np.ones(nh)
rhl = RBFRandomHiddenLayer(n_hidden=nh, activation_func='poly_spline', gamma=3)
#rhl = RBFRandomHiddenLayer(n_hidden=nh, activation_func='multiquadric', gamma=1)
#rhl = RBFRandomHiddenLayer(n_hidden=nh, centers=ctrs, radii=unit_rs, gamma=4)
elmr = ELMRegressor(hidden_layer=rhl)
elmr.fit(xtoy_train, ytoy_train)
print elmr.score(xtoy_train, ytoy_train), elmr.score(xtoy_test, ytoy_test)
plot(xtoy, ytoy, xtoy, elmr.predict(xtoy))

# <codecell>

rbf_rhl = RBFRandomHiddenLayer(n_hidden=100, random_state=0, gamma=0.1)
elmc_rbf = ELMClassifier(hidden_layer=rbf_rhl)
elmc_rbf.fit(dgx_train, dgy_train)
print elmc_rbf.score(dgx_train, dgy_train), elmc_rbf.score(dgx_test, dgy_test)

def powtanh_xfer(activations, power=1.0):
    return pow(np.tanh(activations), power)

#tanh_rhl = SimpleRandomHiddenLayer(n_hidden=5000, random_state=0)
tanh_rhl = SimpleRandomHiddenLayer(n_hidden=5000, activation_func=powtanh_xfer, activation_args={'power':2.0})
elmc_tanh = ELMClassifier(hidden_layer=tanh_rhl)
elmc_tanh.fit(dgx_train, dgy_train)
print elmc_tanh.score(dgx_train, dgy_train), elmc_tanh.score(dgx_test, dgy_test)

# <codecell>

rbf_rhl = RBFRandomHiddenLayer(n_hidden=100, gamma=0.1)
tr, ts = res_dist(dgx, dgy, ELMClassifier(hidden_layer=rbf_rhl), n_runs=100, random_state=0)

# <codecell>

hist(ts), hist(tr)
print

# <codecell>

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
tr, ts = res_dist(dbx, dby, RandomForestRegressor(n_estimators=15), n_runs=100, random_state=0)
hist(tr), hist(ts)
print

rhl = RBFRandomHiddenLayer(n_hidden=15, gamma=0.01)
tr,ts = res_dist(dbx, dby, ELMRegressor(rhl), n_runs=100, random_state=0)
hist(tr), hist(ts)
print

# <codecell>

hist(ts), hist(tr)
print

# <codecell>

elmc = SimpleELMClassifier(n_hidden=500)
elmc.fit(dgx_train, dgy_train)
print elmc.score(dgx_train, dgy_train), elmc.score(dgx_test, dgy_test)

# <codecell>

elmc = SimpleELMClassifier(n_hidden=500, activation_func='hardlim')
elmc.fit(dgx_train, dgy_train)
print elmc.score(dgx_train, dgy_train), elmc.score(dgx_test, dgy_test)

# <codecell>

elmr = SimpleELMRegressor()
elmr.fit(xtoy_train, ytoy_train)
print elmr.score(xtoy_train, ytoy_train), elmr.score(xtoy_test, ytoy_test)
plot(xtoy, ytoy, xtoy, elmr.predict(xtoy))

# <codecell>

elmr = SimpleELMRegressor(activation_func='tribas')
elmr.fit(xtoy_train, ytoy_train)
print elmr.score(xtoy_train, ytoy_train), elmr.score(xtoy_test, ytoy_test)
plot(xtoy, ytoy, xtoy, elmr.predict(xtoy))

