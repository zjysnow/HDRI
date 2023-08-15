import numpy as np

def getLut2(func, lut_bit = 24, step_bit1 = 4, step_bit2 = 6, bvalue = 128):
    input_bit = 12
    lut_x1 = np.array(range(0, bvalue+(1<<step_bit1), 1<<step_bit1))/((1<<input_bit)-1)
    lut_x2 = np.array(range(bvalue+(1<<step_bit2), (1<<input_bit)+(1<<step_bit2), 1<<step_bit2))/((1<<input_bit)-1)

    lut1 = np.round(func(lut_x1) * ((1<<lut_bit)-1)).astype(np.uint64)
    lut2 = np.round(func(lut_x2) * ((1<<lut_bit)-1)).astype(np.uint64)
    lut2 = np.minimum(lut2, (1<<lut_bit)-1)

    return lut1, lut2

def lutEOTF(x, lut1, lut2, lut_bit = 24, step_bit1 = 4, step_bit2 = 6, bvalue = 128):
    x1 = x[x<=bvalue]
    x2 = x[x>bvalue]

    index1 = x1 >> step_bit1
    index2 = (x2-bvalue) >> step_bit2

    resi1 = x1 & ((1<<(step_bit1))-1)
    resi2 = x2 & ((1<<(step_bit2))-1)

    lut_min1 = lut1[index1]
    lut_min2 = np.piecewise(index2, [index2>0], [
        lambda idx: lut2[idx-1],
        lambda idx: lut1[-1],
    ])

    lut_max1 = np.piecewise(index1, [index1<(bvalue>>step_bit1)], [
        lambda idx: lut1[idx+1],
        lambda idx: lut1[-1]
    ])
    lut_max2 = lut2[index2]

    interp_val1 = resi1*lut_max1 + ((1<<step_bit1)-resi1)*lut_min1 + (1<<(step_bit1-1))
    interp_val2 = resi2*lut_max2 + ((1<<step_bit2)-resi2)*lut_min2 + (1<<(step_bit2-1))

    y1 = interp_val1 >> step_bit1
    y2 = interp_val2 >> step_bit2
    y2 = np.minimum(y2, (1<<lut_bit)-1)
    
    y = np.zeros_like(x)
    y[x<=bvalue] = y1
    y[x>bvalue] = y2

    return y