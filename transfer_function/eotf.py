import numpy as np

def PQ(x):
    m1 = 2610 / 16384
    m2 = 2523 / 4096 * 128
    c1 = 3424 / 4096
    c2 = 2413 / 4096 * 32
    c3 = 2392 / 4096 * 32 
    return (np.maximum((x**(1/m2))-c1, 0) / (c2 - c3 * (x**(1/m2)))) ** (1/m1)
