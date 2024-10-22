import numpy as np
import cv2

def load_yuv(file, width, height, type="yuv420p10"):
    if type == "yuv420p10":
        data = np.fromfile(file, dtype=np.uint16).reshape(-1, height*3//2, width)
        frames = []
        for yuv in data:
            Y = yuv[:height, :]
            U = cv2.resize(yuv[height:height*5//4, :].reshape(height//2, width//2), (width, height))
            V = cv2.resize(yuv[height*5//4:, :].reshape(height//2, width//2), (width, height))
            frames.append(np.concatenate((Y[...,np.newaxis],U[...,np.newaxis],V[...,np.newaxis]), axis=2))
        return frames
    else:
        print("unknown type")
    pass