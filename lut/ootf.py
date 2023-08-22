import numpy as np

def interpolate(x, lut, input_bit = 24):
    bit = (input_bit - np.log2(lut.shape[0] - 1)).astype(np.uint64)
    print(bit)
    idx = x >> bit
    x0 = idx << bit
    y0 = lut[idx]
    y1 = lut[idx+1]
    gain = (((y1-y0)*(x-x0)).astype(np.uint64) >> bit) + y0
    return gain

def lutOOTF(x, gain_lut, lut_bit = 16, input_bit = 24):
    gain = interpolate(x, gain_lut, input_bit)
    rounding = 1<<(lut_bit-1)
    print(gain)
    y = (x * gain + rounding).astype(np.uint64) >> lut_bit
    y = np.clip(y, 0, (1<<input_bit)-1)
    return y

if __name__ == "__main__":
    lut = (np.array(range(257))/256*((1<<16)-1)).astype(np.uint64)
    print(lut)
    x = np.round(np.random.rand(3,3) * ((1<<24)-1)).astype(np.uint64)
    y = lutOOTF(x, lut)
    print(y)