import colour
import numpy as np
from scipy.optimize import fmin
from colour.utilities import as_float

settings_BT2246 = {
    "k": 3.0,
    "T": 0.1,
    "X_max": 12,
    "N_max": 15,
    "n": 0.03,
    "p": 1.2274 * 10**6,
    "phi_0": 3 * 10**-8,
    "u_0": 7,
}

def maximise_spatial_frequency(L):
    maximised_spatial_frequency = []
    for L_v in L:
        X_0 = 60
        d = colour.contrast.pupil_diameter_Barten1999(L_v, X_0)
        sigma = colour.contrast.sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
        E = colour.contrast.retinal_illuminance_Barten1999(L_v, d, True)
        maximised_spatial_frequency.append(
                 fmin(
                 lambda x: (
                        -colour.contrast.contrast_sensitivity_function_Barten1999(
                        u=x,
                        sigma=sigma,
                        X_0=X_0,
                        E=E,
                        **settings_BT2246
                     )
                 ),
                 0,
                 disp=False,
             )[0]
         )
    return as_float(np.array(maximised_spatial_frequency))

def barten_ramp():
    L = np.logspace(np.log10(0.0001), np.log10(10000), 10)
    X_0 = Y_0 = 60
    d = colour.contrast.pupil_diameter_Barten1999(L, X_0, Y_0)
    sigma = colour.contrast.sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
    E = colour.contrast.retinal_illuminance_Barten1999(L, d)
    u = maximise_spatial_frequency(L)

    ramp = 2 / colour.contrast.contrast_sensitivity_function_Barten1999(
            u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0, **settings_BT2246
        )  / 0.64
    
    return L, ramp

def barten_flat():
    L = np.logspace(np.log10(0.0001), np.log10(10000), 10)
    X_0 = Y_0 = 60
    d = colour.contrast.pupil_diameter_Barten1999(L, X_0, Y_0)
    sigma = colour.contrast.sigma_Barten1999(0.5 / 60, 0.08 / 60, d)
    E = colour.contrast.retinal_illuminance_Barten1999(L, d)
    u = maximise_spatial_frequency(L)

    flat = 2 / colour.contrast.contrast_sensitivity_function_Barten1999(
            u=u, sigma=sigma, E=E, X_0=X_0, Y_0=Y_0, **settings_BT2246
        ) * (1/1.27) # np.pi/4
    
    return L, flat
    