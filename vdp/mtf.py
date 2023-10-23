import numpy as np
from .otf_cie99 import otf_cie99
from .metric_params import MetricParams

def mtf(rho, metric_par: MetricParams):
    if metric_par.mtf == "cie" or metric_par.mtf == "cie99":
        MTF = otf_cie99(rho, metric_par.age)
    elif metric_par.mtf == "hdrvdp":
        MTF = np.zeros_like(rho)
        for i in range(4):
            MTF = MTF + metric_par.mtf_params_a[i] * np.exp(-metric_par.mtf_params_b[i] * rho)
    else:
        TypeError
    return MTF


if __name__ == "__main__":
    metric_par = MetricParams()
    MTF = mtf(np.array([1,2,3]), metric_par)
    print(MTF)
    pass