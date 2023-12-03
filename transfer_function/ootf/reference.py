import numpy as np

def HLG(E, Y, Lw = 1000, Lamb = 5):
    '''
    Lw = 400nit - 2000nit
    Lamb >= 5nit
    '''
    gamma = 1.2 + 0.42 * np.log10(Lw/1000) - 0.076*np.log10(Lamb/5)
    return (Y**(gamma-1)) * E 

def HLG_extend(E, Y, Lw = 1000, Lamb = 5):
    '''
    Lw may >= 2000nit
    Lamb >= 5nit
    '''
    gamma = 1.2 * (1.111**np.log2(Lw/1000)) * (0.98**(np.log2(Lamb/5)))
    return (Y**(gamma-1)) * E 


def PQ(E):
    '''
    OOTF_sdr = EOTF_1886[OETF_709]
    E [0,1]
    out: [0,10000]
    '''
    Ep = np.piecewise(
        E, [E < 0.018/59.5208], [
        lambda x: 4.5*(59.5208*x),
        lambda x: 1.099*((59.5208*x)**0.45)-0.099
        ]
    )
    return 100*Ep**2.4