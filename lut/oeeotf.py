import numpy as np
import matplotlib.pyplot as plt

def getLut2(func, input_bit = 12, lut_bit = 24, step_bit1 = 4, step_bit2 = 6, bvalue = 128):
    # input_bit = 12
    lut_x1 = np.array(range(0, bvalue+(1<<step_bit1), 1<<step_bit1))/((1<<input_bit)-1)
    lut_x2 = np.array(range(bvalue+(1<<step_bit2), (1<<input_bit)+(1<<step_bit2), 1<<step_bit2))/((1<<input_bit)-1)

    lut1 = np.round(func(lut_x1) * ((1<<lut_bit)-1)).astype(np.uint64)
    lut2 = np.round(func(lut_x2) * ((1<<lut_bit)-1)).astype(np.uint64)
    lut2 = np.minimum(lut2, (1<<lut_bit)-1)

    return lut1, lut2

def lutEOTF(x, lut1, lut2, lut_bit:int = 24, step_bit1:int = 4, step_bit2:int = 6, bvalue:int = 128):
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

    y1 = np.int64(interp_val1) >> step_bit1
    y2 = np.int64(interp_val2) >> step_bit2
    y2 = np.minimum(y2, (1<<lut_bit)-1)
    
    y = np.zeros_like(x, dtype=np.uint64)
    y[x<=bvalue] = y1
    y[x>bvalue] = y2

    return y

def getLutIndex(x, lut1, lut2, lut1_max):
    return np.piecewise(x, [x >= lut1_max], [
        lambda x: [np.sum(lut2<x2) for x2 in x],
        lambda x: [np.sum(lut1<x1) for x1 in x]
    ])

# def getLutIndex(x, lut):
#     return np.array([np.sum(lut<x1) for x1 in x])

def lutOETF(x, lut1, lut2, lut_bit = 24, step_bit1 = 4, step_bit2 = 6, bvalue:int = 128, output_bit = 12):

    lut1_max = lut1[-1]
    index = getLutIndex(x, lut1, lut2, lut1_max)

    index1_max = lut1.shape[0]
    index2_max = lut2.shape[0]
    lut_min = np.piecewise(index, [(x<lut1_max)&(index==0), (x<lut1_max)&(index>index1_max), 
                                   (x<lut1_max)&(index<=index1_max)&(index>0), 
                                   (x>=lut1_max)&(index==0), (x>=lut1_max)&(index>0)], [
        lambda x: 0,
        lambda x: lut1[-1],
        lambda x: lut1[x-1],
        lambda x: lut1[-1],
        lambda x: lut2[x-1]
    ])

    lut_max = np.piecewise(index, [(x<lut1_max)&(index>(index1_max-1)), (x<lut1_max)&(index<=(index1_max-1)), 
                                   (x>=lut1_max)&(index>=index2_max), (x>=lut1_max)&(index<index2_max)], [
        lambda x: (1<<lut_bit)-1,
        lambda x: lut1[x],
        lambda x: (1<<lut_bit)-1,
        lambda x: lut2[x]
    ])

    lut_base = np.piecewise(index, [(x<lut1_max)&(index==0), (x<lut1_max)&(index>0), 
                                    (x>=lut1_max)&(index>=index2_max), (x>=lut1_max)&(index<index2_max)], [
        lambda x: 0,
        lambda x: (x - 1) << step_bit1,
        lambda x: (1<<output_bit)-1,
        lambda x: (x + ((1<<output_bit)>>step_bit2) - lut2.shape[0]) << step_bit2
    ])

    step = (lut_max - lut_min).astype(np.int64)
    resi = x - lut_min
    resi[x<lut1_max] <<= (step_bit1 + 1)
    resi[x>=lut1_max] <<= (step_bit2 + 1)
    
    resi_carry = (resi + step - 1).astype(np.int64)
    resi_carry[step==0] = 0

    level = np.zeros_like(x)
    level[step!=0] = np.minimum(resi_carry[step!=0] / step[step!=0], bvalue)
    print(lut1_max)

    inc = level >> 1
    inc[(x<lut1_max)&(level>((1<<(step_bit1+1))-1))] = (1<<step_bit1)
    
    y = np.minimum(lut_base + inc, (1<<output_bit)-1)
    return y

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # lut1, lut2 = getLut2(lambda x: x**2.2)

    # x = np.random.randint(0,4095, (3,3), dtype=np.uint64)
    # print(x)

    # y = lutEOTF(x, lut1, lut2)

    # xp = lutOETF(y, lut1, lut2)

    # print(xp)

    # print(np.sum((x - xp)**2))
    
