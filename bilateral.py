import cv2
import numpy as np

def gauss_ker(k, sig):
    x = np.linspace(-(k//2), (k//2), k)
    gx, gy = np.meshgrid(x, x)
    kernel = np.exp(-1*(gx**2 + gy**2)/(2*(sig**2)))
    return kernel

def bilateral(imflash, imambient):
    sigmag = 1
    ws = 11
#     sigmab1 = (np.max(imambient) - np.min(imambient))*(255/10)
    sigmab1 = 1
    gauss_mask = gauss_ker(ws, sigmag)
    
    bias = (ws//2)
    flashpad = np.lib.pad(imflash, (bias, bias), 'edge')
    ambientpad = np.lib.pad(imambient, (bias, bias), 'edge')
    
    h, w = imflash.shape
    Ajoint = np.zeros((h, w))
    Abase = np.zeros((h, w))
    Fbase = np.zeros((h, w))
    
    for i in range(bias, h+bias):
        for j in range(bias, w+bias):
            amb_mask = ambientpad[i-bias:i+bias+1, j-bias:j+bias+1]
            flash_mask = flashpad[i-bias:i+bias+1, j-bias:j+bias+1]
            flash_diffmask = flash_mask - flashpad[i, j]
            amb_diffmask = amb_mask - ambientpad[i, j]
            bil_mask_flash = np.exp(-1*((flash_diffmask/sigmab1)**2)/(2*(sigmab1**2)))
            bil_mask_amb = np.exp(-1*((amb_diffmask/sigmab1)**2)/(2*(sigmab1**2)))
            filt_mask_flash = bil_mask_flash*gauss_mask
            norm_term_flash = np.sum(filt_mask_flash)
            filt_mask_amb = bil_mask_amb*gauss_mask
            norm_term_amb = np.sum(filt_mask_amb)
            Ajoint_mask = (amb_mask*filt_mask_flash)/norm_term_flash
            Abase_mask = (amb_mask*filt_mask_amb)/norm_term_amb
            Fbase_mask = (flash_mask*filt_mask_flash)/norm_term_flash
            Ajoint[i-bias, j-bias] = np.sum(Ajoint_mask)
            Abase[i-bias, j-bias] = np.sum(Abase_mask)
            Fbase[i-bias, j-bias] = np.sum(Fbase_mask)
            
    
    return [Ajoint, Abase, Fbase]

