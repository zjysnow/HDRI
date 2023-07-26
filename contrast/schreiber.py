import numpy as np

def schreiber_limit():
    '''
    the slope is not accurate
    '''
    L = np.logspace(-3,3,100)
    limit = np.piecewise(L, [L<1, L>=1], [
        lambda x: 0.02*np.power(x,-0.302),
        lambda x: 0.02
    ])
    return L, limit