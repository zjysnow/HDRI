import numpy as np
from .metric_params import MetricParams
from .utils import load_spectral_resp, pix_per_degree, ncsf
from .multscale import MultScalLPYR
from .virtual_pathway import visual_pathway
from .civdm import civdm
from .csf import CSF

def hdrvdp3(task, test, reference, color_encoding, ppd):
    metric_par = MetricParams()

    img_channels = test.shape[2]

    metric_par.pad_value = 0 # determine_padding(reference, metric_par)

    metric_par.pix_per_deg = ppd

    _, IMG_E = load_spectral_resp("vdp/data/emission_spectra_led-lcd-srgb.csv")
    if img_channels == 1 and IMG_E.shape[1] > 1:
        IMG_E = np.sum(IMG_E, 1)

    if img_channels != IMG_E.shape[1]:
        ValueError
    
    metric_par.spectral_emission = IMG_E

    # should translate color space
    # for example all to P3
    

    metric_par.mult_scale = MultScalLPYR()

    B_R, L_adapt_reference,  bb_padvalue, P_ref = visual_pathway(reference, "reference", metric_par, -1)
    B_T, L_adapt_test, bb_padvalue, P_test = visual_pathway(test, "test", metric_par, bb_padvalue)

    print("test max: ", test.max())
    print("B_T.get_band(0).max: ", B_T.get_band(0).max())

    band_freq= B_T.get_freqs()
    print("band_ferq: ", band_freq)
    # precompute CSF
    csf = CSF()
    # csf_la = np.logspace(-5,5,256)
    # csf_log_la = np.log10(csf_la)
    csf.S = np.zeros((256, B_T.band_count()))

    for b in range(B_T.band_count()):
        csf.S[:,b] = ncsf(band_freq[b], csf.csf_la.T, metric_par)
    # print(S)
    # plt.plot(csf.S)
    # plt.show()
    # L_mean_adapt = (L_adapt_test+ L_adapt_reference)/2
    # log_La = np.log10(np.clip(L_mean_adapt, csf.csf_la[0], csf.csf_la[-1]))

    res = civdm(B_T, L_adapt_test, B_R, L_adapt_reference, csf, metric_par)

    # print(res[0].max(), res[0].min())
    # plt.imshow(res[0])
    # plt.show()
    return res

if __name__ == "__main__":

    # Y_tonemapped = np.random.randn(112,112,3)
    # I_hdr = np.random.randn(112,112,3) * 10
    Y_tonemapped = 10
    I_hdr = 10

    ppd = pix_per_degree(30, [112,112], 0.5)

    hdrvdp3("civdm", Y_tonemapped, I_hdr, "rgb-native", ppd)
