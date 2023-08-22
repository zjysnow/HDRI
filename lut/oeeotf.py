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
    
    y = np.zeros_like(x, dtype=np.uint64)
    y[x<=bvalue] = y1
    y[x>bvalue] = y2

    return y

def getLutIndex(x, lut1, lut2, bvalue):
    return np.piecewise(x, [x >= bvalue], [
        lambda x: [np.sum(lut2<x2) for x2 in x],
        lambda x: [np.sum(lut1<x1) for x1 in x]
    ])

# def getLutIndex(x, lut):
#     return np.array([np.sum(lut<x1) for x1 in x])

def lutOETF(x, lut1, lut2, lut_bit = 24, step_bit1 = 4, step_bit2 = 6):

    bvalue = lut1[-1]
    index = getLutIndex(x, lut1, lut2, lut1[-1])

    lut_min = np.piecewise(index, [(x<bvalue)&(index==0), (x<bvalue)&(index>9), (x<bvalue)&(index<=9)&(index>0), 
                                   (x>=bvalue)&(index==0), (x>=bvalue)&(index>0)], [
        lambda x: 0,
        lambda x: lut1[-1],
        lambda x: lut1[x-1],
        lambda x: lut1[-1],
        lambda x: lut2[x-1]
    ])

    lut_max = np.piecewise(index, [(x<bvalue)&(index>8), (x<bvalue)&(index<=8), 
                                   (x>=bvalue)&(index>=62), (x>=bvalue)&(index<62)], [
        lambda x: (1<<lut_bit)-1,
        lambda x: lut1[x],
        lambda x: (1<<lut_bit)-1,
        lambda x: lut2[x]
    ])

    lut_base = np.piecewise(index, [(x<bvalue)&(index==0), (x<bvalue)&(index>0), 
                                    (x>=bvalue)&(index>62), (x>=bvalue)&(index<62)], [
        lambda x: 0,
        lambda x: (x - 1) << 4,
        lambda x: 4095,
        lambda x: (x + 2) << 6
    ])

    step = lut_max - lut_min
    resi = x - lut_min
    resi[x<bvalue] <<= (step_bit1 + 1)
    resi[x>=bvalue] <<= (step_bit2 + 1)
    
    resi_carry = resi + step - 1
    resi_carry[step==0] = 0

    level = np.zeros_like(x)
    level[step!=0] = np.minimum(resi_carry[step!=0] / step[step!=0], 128)

    inc = level >> 1
    inc[(x<bvalue)&(level>31)] = 16
    
    y = np.minimum(lut_base + inc, 4095)
    return y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lut1, lut2 = getLut2(lambda x: x**2.2)

    x = np.array(range(4096), dtype=np.uint64)
    y = lutEOTF(x, lut1, lut2)

    v = lutOETF(y, lut1, lut2)

    # plt.plot(x/4095, x/4095, "--")
    # plt.plot(x/4095., y/16777215.)
    plt.figure()
    plt.plot(x, v)

    # x = lutOETF(y, lut1, lut2)
    # plt.plot(x/4095, y)

    plt.show()
    
