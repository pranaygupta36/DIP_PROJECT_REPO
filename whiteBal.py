import cv2
import numpy as np

def whiteBal(imambient, imflash, opt, alpha):
    if opt == 1:
        scaling = np.array([255/246, 255/169, 255/87])
        imambient[:,:,0] = imambient[:,:,0]*scaling[0]
        imambient[:,:,1] = imambient[:,:,1]*scaling[1]
        imambient[:,:,2] = imambient[:,:,2]*scaling[2]
        wb = imambient
    else:
        [h,w,_] = imflash.shape
        imambientr = imambient[:,:,0]
        imambientg = imambient[:,:,1]
        imambientb = imambient[:,:,2]

        linflashr = cv2.equalizeHist(imflash[:,:,0])
        linambientr = cv2.equalizeHist(imambientr)
        linflashg = cv2.equalizeHist(imflash[:,:,1])
        linambientg = cv2.equalizeHist(imambientg)
        linflashb = cv2.equalizeHist(imflash[:,:,2])
        linambientb = cv2.equalizeHist(imambientb)

        albedor = linflashr - linambientr
        albedog = linflashg - linambientg
        albedob = linflashb - linambientb

        imambientr = imambientr.astype('double')/255
        imambientg = imambientg.astype('double')/255
        imambientb = imambientb.astype('double')/255
        
        albedor = albedor.astype('double')/255
        albedog = albedog.astype('double')/255
        albedob = albedob.astype('double')/255
        
        thr1r = 0.02*(np.max(imambientr) - np.min(imambientr))
        thr1g = 0.02*(np.max(imambientg) - np.min(imambientg))
        thr1b = 0.02*(np.max(imambientb) - np.min(imambientb))

        thr2r = 0.02*(np.max(albedor) - np.min(albedor))
        thr2g = 0.02*(np.max(albedog) - np.min(albedog))
        thr2b = 0.02*(np.max(albedob) - np.min(albedob))

        albedor[albedor == 0] = 0.0001
        albedog[albedog == 0] = 0.0001
        albedob[albedob == 0] = 0.0001
        
        Cr = imambientr/albedor
        Cg = imambientg/albedog
        Cb = imambientb/albedob

        mr=0
        mg=0
        mb=0
        Crmean=0
        Cgmean=0
        Cbmean=0;
        
        for i in range(0,h):
            for j in range(0,w):
                if (abs(imambientr[i, j]) > thr1r) and (albedor[i, j] > thr2r):
                    Crmean = Crmean + Cr[i,j]
                    mr = mr + 1
                if (abs(imambientg[i,j]) > thr1g) and (albedog[i,j] > thr2g):
                    Cgmean = Cgmean + Cg[i, j]
                    mg = mg + 1
                if (abs(imambientb[i,j])>thr1b) and (albedob[i,j] > thr2b):
                    Cbmean = Cbmean + Cb[i,j]
                    mb=mb + 1
                    
        Crmean = Crmean/mr
        Cgmean = Cgmean/mg
        Cbmean = Cbmean/mb

        wb = np.dstack([(imambientr/Crmean)*alpha, (imambientg/Cgmean)*alpha, (imambientb/Cbmean)*alpha])
        wb[wb>1] = 1
        
    return wb