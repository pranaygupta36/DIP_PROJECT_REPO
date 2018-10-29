import cv2
import numpy as np

def flashAdj(ima, imf, alpha ):
    ya = cv2.cvtColor(ima, cv2.COLOR_BGR2YCR_CB)
    yf = cv2.cvtColor(imf, cv2.COLOR_BGR2YCR_CB)
    im = (1-alpha)*ya + alpha*yf;
    im = cv2.cvtColor(im, cv2.COLOR_YCR_CB2RGB)

    return im