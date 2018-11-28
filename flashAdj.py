import cv2
import numpy as np

def flashAdj(ima, imf, alpha):
    ya = cv2.cvtColor(ima, cv2.COLOR_BGR2YCR_CB)
    yf = cv2.cvtColor(imf, cv2.COLOR_BGR2YCR_CB)
    im = np.zeros(ya.shape).astype('double')
#     if alpha>1:
#         im[:,:,0] = alpha*ya[:,:,0] + (1-alpha)*yf[:,:,0]
#         im[:,:,1] = yf[:,:,1]
#         im[:,:,2] = yf[:,:,2]
#     elif alpha<0:
#         im[:,:,0] = alpha*ya[:,:,0] + (1-alpha)*yf[:,:,0]
#         im[:,:,1] = ya[:,:,1]
#         im[:,:,2] = ya[:,:,2]
#     else:
    im = alpha*ya + (1-alpha)*yf
    im[im>255] = 255
    im[im<0] = 0
    im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_YCR_CB2RGB)
    return im