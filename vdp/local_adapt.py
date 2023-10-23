import numpy as np
from .fast_gauss import fast_gauss

def local_adapt(L_otf, ppd):
    sigma = 10**-0.781367 * ppd
    L_la = np.exp(fast_gauss(np.log(np.maximum(L_otf, 1e-6)), sigma))
    return L_la