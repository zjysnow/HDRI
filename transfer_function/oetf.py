import numpy as np

def PQ(x):
    m1 = 2610 / 16384
    m2 = 2523 / 4096 * 128
    c1 = 3424 / 4096
    c2 = 2413 / 4096 * 32
    c3 = 2392 / 4096 * 32 
    return ((c1+c2*(x**m1))/(1+c3*(x**m1)))**m2

def HLG(x):
    a = 0.17883277
    b = 1-4*a
    c = 0.5-a*np.log(4*a)
    return np.piecewise(x, [x<=1/12], [
        lambda x: np.sqrt(3*x),
        lambda x: a*np.log(12*x-b)+c
    ])

def sRGB(x):
    return np.piecewise(
        x, [x < 0.00304], [ 
        lambda x: 12.92*x, # x < 0.00304 
        lambda x: 1.055 * x**(1/2.4) - 0.055 # else
        ]
    )

def BT709(x):
    return np.piecewise(
        x, [x < 0.018], [
        lambda x: 4.5*x,
        lambda x: 1.099*(x**0.45)-0.099
        ]
    )