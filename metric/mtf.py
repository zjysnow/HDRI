import numpy as np
from scipy.special import diric, kv

class MTF:
    def __init__(self, method = "hdrvdp", age = 24, p = 0.5):
        self.method = method

        par = [0.061466549455263, 0.99727370023777070]; # old parametrization of MTF
        self.mtf_params_a = np.array([par[1]*0.426, par[1]*0.574, (1-par[1])*par[0], (1-par[1])*(1-par[0])])
        self.mtf_params_b = np.array([0.028, 0.37, 37, 360])

        self.age = age
        self.p = p
        pass

    def __call__(self, rho):
        if self.method == "hdrvdp":
            return self.hdrvdp(rho)
        if self.method == "cie99":
            return self.otf_cie99(rho)
        NotImplemented

    def hdrvdp(self, rho):
        MTF = np.zeros_like(rho)
        for i in range(4):
            MTF = MTF + self.mtf_params_a[i] * np.exp(-self.mtf_params_b[i] * rho)
        return MTF
    
    def otf_cie99(self, rho):
        def cie_mtf(omega, age, p):
            # % The equation was found by applying a Fourier transform using Matlab's
            # % symbolic toolbox

            c1=9.2e6
            c2=0.08
            c3=0.0046
            c4=1.5e5
            c5=0.045
            c6=1.6
            c7=400
            c8=0.1
            c9=3e-8
            c10=1300
            c11=0.8
            c12=2.5e-3
            c13=0.0417
            c14=0.055

            age_m=70

            M = (p*((age**4*c6)/age_m**4 + 1)*(2*c8*c11*kv(0, c8*np.abs(omega)) + 2*c8**2*c10*np.abs(omega)*kv(1, c8*np.abs(omega)))
                - ((age**4*c2)/age_m**4 - 1)*(2*c1*c3**2*np.abs(omega)*kv(1, c3*np.abs(omega)) + 2*c4*c5**2*np.abs(omega)*kv(1, c5*np.abs(omega)))
                + 2*np.pi*c12*p*diric(omega, 1) - (2*c9*np.pi*(c6*age**4 + age_m**4)*diric(omega,2))/age_m**4 + (c7*c8*np.pi*np.exp(-c8*np.abs(omega))*(c6*age**4 + age_m**4))/age_m**4)/(p*(c13 + (age**4*c14)/age_m**4) + 1)
            return M
        
        omega = 2 * np.pi * rho
        M = np.piecewise(omega, [omega==0], [
            lambda x: 1,
            lambda x:cie_mtf(x, self.age, self.p) / cie_mtf(0.0001, self.age, self.p)
        ])
        return M
    

