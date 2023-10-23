import numpy as np
from .laplacian_pyramid import laplacian_pyramid
import copy

class MultScale():
    def decompose(self, I, ppd):
        NotImplemented

    def reconstruct(self):
        NotImplemented

    def get_band(self, band, o):
        NotImplemented

    def set_band(self, band, o, B):
        NotImplemented

    def band_count(self):
        NotImplemented

    def orient_count(self, band):
        NotImplemented

    def get_freqs(self):
        NotImplemented


class MultScalLPYR(MultScale):
    def __init__(self):
        super().__init__()

    def decompose(self, I, ppd):
        self.ppd = ppd
        self.img_sz = I.shape

        height = int(np.maximum( np.ceil(np.log2(ppd))-2, 1))
        
        self.band_freqs = np.ones(shape=int(height+1), dtype=np.float32)
        self.band_freqs[1:] = 0.3228*(2**-np.linspace(0, height-1, int(height)))
        self.band_freqs *= self.ppd/2
        
        self.P,_ = laplacian_pyramid(I, height+1) # P is N H W array
        return copy.deepcopy(self)

    def reconstruct(self):
        I = np.zeros((self.P.shape[1], self.P.shape[2]))
        for i in range(self.P.shape[0]):
            # print(self.P[i].max(), self.P[i].min())
            I = I + self.P[i]
        return I.copy()

    def get_band(self, band):
        return np.ascontiguousarray(self.P[band])

    def set_band(self, band, B):
        self.P[band] = np.ascontiguousarray(B)
        return self

    def band_count(self):
        return self.P.shape[0]
    
    def band_size(self):
        return self.P.shape[1], self.P.shape[2]

    def orient_count(self, band):
        return 1

    def get_freqs(self):
        return self.band_freqs

    
if __name__ == "__main__":
    ppd = 52.722441427310380
    height = np.maximum(np.ceil(np.log2(ppd))-2, 1)
    
    band_freqs = np.ones(shape=int(height+1), dtype=np.float32)
    band_freqs[1:] = 0.3228*(2**-np.linspace(0, height-1, int(height)))

    band_freqs *= ppd/2

    print(band_freqs)