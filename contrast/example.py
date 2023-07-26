from barten import barten_flat, barten_ramp
from schreiber import schreiber_limit

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    L, flat = barten_flat()
    plt.loglog(L, flat, "--")
    
    L, ramp = barten_ramp()
    plt.loglog(L, ramp, "--")

    L, limit = schreiber_limit()
    plt.loglog(L, limit, "--")

    x = np.linspace(0,1,256)
    L = 100*(x**2.2)+0.5
    dL = L[1:] - L[:-1]
    plt.loglog(L[:-1], dL/L[:-1], label="gamma2.2 8bit with 5% LP 0.5-100nit") 

    L = 100*(x**2.2)
    L = L[L>=0.5]
    dL = L[1:] - L[:-1]
    plt.loglog(L[:-1], dL/L[:-1], label="gamma2.2 8bit 0.5-100nit") 

    plt.ylabel("deltaL/L")
    plt.xlabel("Luminance (cd/m2)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
