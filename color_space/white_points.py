import numpy as np

A    = 2856
B    = 4878
C    = 6774
D50  = 5003
D65  = 6504
D75  = 7504
D93  = 9305
E    = 5454
ACSE = 5998
DCI = 6304
 

WhitePoint = {
    # reference https://en.wikipedia.org/wiki/Template:Color_temperature_white_points
    2856: np.array([0.44757,    0.40745]), # A
    4878: np.array([0.34842,    0.35161]), # B
    6774: np.array([0.31006,    0.31616]), # C / NTSC
    5003: np.array([0.34567,    0.35850]), # D50
    6304: np.array([0.314,      0.351]),   # SMPTE 432
    6504: np.array([0.31271,    0.32902]), # CIE D65 average daylight
    7504: np.array([0.29902,    0.31485]), # D75 north sky daylight
    9305: np.array([0.28315,    0.29711]), # D93, BT2035
    5454: np.array([0.33333,    0.33333]), # E equal energy

    5998: np.array([0.32168,    0.33767]), # ACSE
    
    # reference http://www.vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.html
    1000:	np.array([0.6499,	0.3474]),
    1100:	np.array([0.6361,	0.3594]),
    1200:	np.array([0.6226,	0.3703]),
    1300:	np.array([0.6095,	0.3801]),
    1400:	np.array([0.5966,	0.3887]),
    1500:	np.array([0.5841,	0.3962]),
    1600:	np.array([0.572,	0.4025]),
    1700:	np.array([0.5601,	0.4076]),
    1800:	np.array([0.5486,	0.4118]),
    1900:	np.array([0.5375,	0.415]),
    2000:	np.array([0.5267,	0.4173]),
    2100:	np.array([0.5162,	0.4188]),
    2200:	np.array([0.5062,	0.4196]),
    2300:	np.array([0.4965,	0.4198]),
    2400:	np.array([0.4872,	0.4194]),
    2500:	np.array([0.4782,	0.4186]),
    2600:	np.array([0.4696,	0.4173]),
    2700:	np.array([0.4614,	0.4158]),
    2800:	np.array([0.4535,	0.4139]),
    2900:	np.array([0.446,	0.4118]),
    3000:	np.array([0.4388,	0.4095]),
    3100:	np.array([0.432,	0.407]),
    3200:	np.array([0.4254,	0.4044]),
    3300:	np.array([0.4192,	0.4018]),
    3400:	np.array([0.4132,	0.399]),
    3500:	np.array([0.4075,	0.3962]),
    3600:	np.array([0.4021,	0.3934]),
    3700:	np.array([0.3969,	0.3905]),
    3800:	np.array([0.3919,	0.3877]),
    3900:	np.array([0.3872,	0.3849]),
    4000:	np.array([0.3827,	0.382]),
    4100:	np.array([0.3784,	0.3793]),
    4200:	np.array([0.3743,	0.3765]),
    4300:	np.array([0.3704,	0.3738]),
    4400:	np.array([0.3666,	0.3711]),
    4500:	np.array([0.3631,	0.3685]),
    4600:	np.array([0.3596,	0.3659]),
    4700:	np.array([0.3563,	0.3634]),
    4800:	np.array([0.3532,	0.3609]),
    4900:	np.array([0.3502,	0.3585]),
    5000:	np.array([0.3473,	0.3561]),
    5100:	np.array([0.3446,	0.3538]),
    5200:	np.array([0.3419,	0.3516]),
    5300:	np.array([0.3394,	0.3494]),
    5400:	np.array([0.3369,	0.3472]),
    5500:	np.array([0.3346,	0.3451]),
    5600:	np.array([0.3323,	0.3431]),
    5700:	np.array([0.3302,	0.3411]),
    5800:	np.array([0.3281,	0.3392]),
    5900:	np.array([0.3261,	0.3373]),
    6000:	np.array([0.3242,	0.3355]),
    6100:	np.array([0.3223,	0.3337]),
    6200:	np.array([0.3205,	0.3319]),
    6300:	np.array([0.3188,	0.3302]),
    6400:	np.array([0.3171,	0.3286]),
    6500:	np.array([0.3155,	0.327]),
    6600:	np.array([0.314,	0.3254]),
    6700:	np.array([0.3125,	0.3238]),
    6800:	np.array([0.311,	0.3224]),
    6900:	np.array([0.3097,	0.3209]),
    7000:	np.array([0.3083,	0.3195]),
    7100:	np.array([0.307,	0.3181]),
    7200:	np.array([0.3058,	0.3168]),
    7300:	np.array([0.3045,	0.3154]),
    7400:	np.array([0.3034,	0.3142]),
    7500:	np.array([0.3022,	0.3129]),
    7600:	np.array([0.3011,	0.3117]),
    7700:	np.array([0.3,	    0.3105]),
    7800:	np.array([0.299,	0.3094]),
    7900:	np.array([0.298,	0.3082]),
    8000:	np.array([0.297,	0.3071]),
    8100:	np.array([0.2961,	0.3061]),
    8200:	np.array([0.2952,	0.305]),
    8300:	np.array([0.2943,	0.304]),
    8400:	np.array([0.2934,	0.303]),
    8500:	np.array([0.2926,	0.302]),
    8600:	np.array([0.2917,	0.3011]),
    8700:	np.array([0.291,	0.3001]),
    8800:	np.array([0.2902,	0.2992]),
    8900:	np.array([0.2894,	0.2983]),
    9000:	np.array([0.2887,	0.2975]),
    9100:	np.array([0.288,	0.2966]),
    9200:	np.array([0.2873,	0.2958]),
    9300:	np.array([0.2866,	0.295]),
    9400:	np.array([0.286,	0.2942]),
    9500:	np.array([0.2853,	0.2934]),
    9600:	np.array([0.2847,	0.2927]),
    9700:	np.array([0.2841,	0.2919]),
    9800:	np.array([0.2835,	0.2912]),
    9900:	np.array([0.2829,	0.2905]),
    10000:	np.array([0.2824,	0.2898]),

    40000: np.array([0.2487,    0.2438]),
}


def getColorTemperature(white_point):
    n = (white_point[0] - 0.3320) / (0.1858 - white_point[1])
    return 437 * (n**3) + 3601 * (n**2) + 6831 * n + 5517

if __name__ == "__main__":
    x = [0.314, 0.351]
    print(getColorTemperature(x))