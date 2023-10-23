import numpy as np

def contrast_sensitivity_function_Daly1993(rho, theta, La, im_size = 1, viewing_dist = 0.5):
    P = 250
    eps = 0.9
    A = 0.801 * (1+0.7/La)**(-0.2)
    B = 0.3 * (1+100/La) ** 0.15

    r_a = 0.856 * viewing_dist ** 0.14
    e = 0 # eccentricity
    r_e = 1 / (1 + 0.24 * e)
    ob = 0.78
    r_theta = (1-ob)/2 * np.cos(4 * theta) + (1+ob)/2 

    B1 = B * eps * rho
    S1 = ((3.23*(rho * rho * im_size)**(-0.3))**5.0 + 1.0) ** (-0.2) * \
        A * eps * rho * np.exp(-B1) * np.sqrt(1+0.06*np.exp(B1))

    rho = rho / (r_a * r_e * r_theta)
    B1 = B * eps * rho
    S2 = ((3.23*(rho * rho * im_size)**(-0.3))**5.0 + 1.0) ** (-0.2) * \
        A * eps * rho * np.exp(-B1) * np.sqrt(1+0.06*np.exp(B1))

    return np.minimum(S1, S2) * P