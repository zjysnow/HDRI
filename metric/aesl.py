import numpy as np

class AESL:
    def __init__(self, aesl_base = -0.125539, aesl_slop_freq = -2.711, age = 24):
        self.aesl_base = aesl_base
        self.aesl_slop_freq = aesl_slop_freq
        self.age = age
        pass

    def __call__(self, rho):
        gamma = 10**self.aesl_base
        S_corr = 10**(-(10**self.aesl_slop_freq * np.log2(rho+gamma)) * np.maximum(0, self.age - 24))
        return S_corr