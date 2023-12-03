import numpy as np
from .FilmicToneCurve import *
from .reference import *

def GTO(x, b, c, dl, dh):
    '''
    four-segment sigmoidal function
    Lp: logarithm of luminance
    b: is the image brightness adjustment parameter
    c: is the contrast parameter
    al, ah: decide on the constrast compression for shadows and highlights
    dl: the lower midtone range 
    dh: the higher midtone range 
    papers: Modeling a Generic Tone-mapping Operator
    '''
    al = (c*dl-1)/dl
    ah = (c*dh-1)/dh
    return np.piecewise(x, [x <=b-dl, (b-dl<x)&(x<=b), (b<x)&(x<=b+dh), b+dh<x], [
        0,
        lambda x: 0.5*c*(x-b)/(1-al*(x-b))+0.5,
        lambda x: 0.5*c*(x-b)/(1+ah*(x-b))+0.5,
        1
    ])