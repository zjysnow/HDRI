import numpy as np
import matplotlib.pyplot as plt

def calc_3dcolor(color, ct_table, input_bit = 12):
    '''
    color h*w*3
    ct_table 17*17*17*3
    '''
    shift_bit = input_bit - 5 + 1 # 17 use 5bits
    rounding = 1 << (shift_bit - 1)
    index = (color >> shift_bit)
    resi = color - (index << shift_bit)
    
    i,j,k = index[:,:,0], index[:,:,1], index[:,:,2]
    t,u,v = resi[:,:,0], resi[:,:,1], resi[:,:,2]

    p0 = ct_table[i,j,k]
    p1 = ct_table[i,j,k+1]
    p2 = ct_table[i,j+1,k]
    p3 = ct_table[i,j+1,k+1]
    p4 = ct_table[i+1,j,k]
    p5 = ct_table[i+1,j,k+1]
    p6 = ct_table[i+1,j+1,k]
    p7 = ct_table[i+1,j+1,k+1]

    # v<t<u # t<u t>v
    mask = (t<u)&(t>v)
    st = p6 - p2
    su = p2 - p0
    sv = p7 - p6

    # t<=v<u  # t<u and t<=v and u<v
    mask = (t<u)&(t<=v)&(u<v)
    st[mask,:] = p7[mask,:] - p3[mask,:]
    su[mask,:] = p3[mask,:] - p1[mask,:]
    sv[mask,:] = p1[mask,:] - p0[mask,:]

    # t<u<=v # t<u t<=v and u>=v
    mask = (t<u)&(t<=v)&(u>=v)
    st[mask,:] = p7[mask,:] - p3[mask,:]
    su[mask,:] = p2[mask,:] - p0[mask,:]
    sv[mask,:] = p3[mask,:] - p2[mask,:]

    # u<=t<v # t>=u t<v
    mask = (t>=u)&(t<v)
    st[mask,:] = p5[mask,:] - p1[mask,:]
    su[mask,:] = p7[mask,:] - p5[mask,:]
    sv[mask,:] = p1[mask,:] - p0[mask,:]

    # v<=u<=t # t>=u t>=v u>=v
    mask = (t>=u)&(t>=v)&(u>=v)
    st[mask,:] = p4[mask,:] - p0[mask,:]
    su[mask,:] = p6[mask,:] - p4[mask,:]
    sv[mask,:] = p7[mask,:] - p6[mask,:]

    # u<v<=t # t>=u t>=v u<v
    mask = (t>=u)&(t>=v)&(u<v)
    st[mask,:] = p4[mask,:] - p0[mask,:]
    su[mask,:] = p7[mask,:] - p5[mask,:]
    sv[mask,:] = p5[mask,:] - p4[mask,:]

    return p0 + ((st*t[:,:,None] + su*u[:,:,None] + sv*v[:,:,None] + rounding).astype(np.int64) >> shift_bit)
    

if __name__ == "__main__":
    ct_table = np.zeros((17,17,17,3), dtype=np.int64)
    for i in range(17):
        for j in range(17):
            for k in range(17):
                ct_table[i,j,k,0] = i<<8
                ct_table[i,j,k,1] = j<<8
                ct_table[i,j,k,2] = k<<8
    # ct_table[0,0,0] = [2048, 2048, 2048]

    color = np.random.randint(0, 4095, (112,112,3), dtype=np.int64)

    # plt.imshow(color/4096)
    # plt.show()

    index = (color >> 8)
    resi = color - (index << 8)
    
    i,j,k = index[:,:,0], index[:,:,1], index[:,:,2]
    t,u,v = resi[:,:,0], resi[:,:,1], resi[:,:,2]

    p0 = ct_table[i,j,k]
    p1 = ct_table[i,j,k+1]
    p2 = ct_table[i,j+1,k]
    p3 = ct_table[i,j+1,k+1]
    p4 = ct_table[i+1,j,k]
    p5 = ct_table[i+1,j,k+1]
    p6 = ct_table[i+1,j+1,k]
    p7 = ct_table[i+1,j+1,k+1]

    # v<t<u # t<u t>v
    mask = (t<u)&(t>v)
    st = p6 - p2
    su = p2 - p0
    sv = p7 - p6

    # t<=v<u  # t<u and t<=v and u<v
    mask = (t<u)&(t<=v)&(u<v)
    st[mask,:] = p7[mask,:] - p3[mask,:]
    su[mask,:] = p3[mask,:] - p1[mask,:]
    sv[mask,:] = p1[mask,:] - p0[mask,:]

    # t<u<=v # t<u t<=v and u>=v
    mask = (t<u)&(t<=v)&(u>=v)
    st[mask,:] = p7[mask,:] - p3[mask,:]
    su[mask,:] = p2[mask,:] - p0[mask,:]
    sv[mask,:] = p3[mask,:] - p2[mask,:]

    # u<=t<v # t>=u t<v
    mask = (t>=u)&(t<v)
    st[mask,:] = p5[mask,:] - p1[mask,:]
    su[mask,:] = p7[mask,:] - p5[mask,:]
    sv[mask,:] = p1[mask,:] - p0[mask,:]

    # v<=u<=t # t>=u t>=v u>=v
    mask = (t>=u)&(t>=v)&(u>=v)
    st[mask,:] = p4[mask,:] - p0[mask,:]
    su[mask,:] = p6[mask,:] - p4[mask,:]
    sv[mask,:] = p7[mask,:] - p6[mask,:]

    # u<v<=t # t>=u t>=v u<v
    mask = (t>=u)&(t>=v)&(u<v)
    st[mask,:] = p4[mask,:] - p0[mask,:]
    su[mask,:] = p7[mask,:] - p5[mask,:]
    sv[mask,:] = p5[mask,:] - p4[mask,:]

    out = p0 + ((st*t[:,:,None] + su*u[:,:,None] + sv*v[:,:,None] + 128).astype(np.int64) >> 8)

    diff = color - out
    print(diff.max(), diff.min())

    x = np.array(range(1024))/1023
    plt.plot(x, x**(1/1.2))
    plt.show()
    
    