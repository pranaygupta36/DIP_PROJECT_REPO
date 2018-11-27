import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import disk
from bilateral import gauss_ker

def shadowRem(imflash, imambient):
    linflash = 0.299*imflash[:,:,0] + 0.587*imflash[:,:,1] + 0.114*imflash[:,:,2]
    linambient = 0.299*imambient[:,:,0] + 0.587*imambient[:,:,1] + 0.114*imambient[:,:,2]
    mask = linflash - linambient
    
    flag = np.zeros((mask.shape), np.uint8)
    thr1 = -0.05
    thr2 = -0.2 
    flag[(mask > thr2) & (mask < thr1)] = 1
    flag[(mask > 0.65) & (mask < 0.7)] = 1
    rang = 0.95*(np.max(linflash) - np.min(linflash))
    flag[linflash > rang] = 1

    se1 = disk(2)
    se2 = disk(6)
    se3 = disk(4)
    flag = cv2.erode(flag, se1, iterations = 1)
    maskff = np.zeros((flag.shape[0]+2, flag.shape[1]+2), np.uint8)
    cv2.floodFill(flag, maskff, (0,0), 1)
    maskff = cv2.dilate(maskff, se2,  iterations = 1)
    maskff = cv2.erode(maskff, se3, iterations = 1)
    maskff = maskff.astype('double')
    k = 3
    sig = 3
    kernel = gauss_ker(3, 3)
    maskff = cv2.filter2D(maskff, -1, kernel)
    
    return maskff