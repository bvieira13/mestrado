

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


dir = 'C:\\Users\\Seminarios CEPOP\\Documents\\Bruno Freitas Vieira\\Imagens\\2022.09.22 - Caracterization\\samples'

names = os.listdir(dir)

I=[]

for name in names:
    I.append(cv.imread(dir+'\\'+name, -1))

I = np.asarray(I)


#%%

dir2 = 'C:\\Users\\Seminarios CEPOP\\Meu Drive\\College\\Biophotonics Lab\\Research\\Programs\\Python\\Camera & Image'

os.chdir(dir2)

import imfun


I2, points = imfun.crop_poly_multiple(I)
I2 = np.asarray(I2)

#%%

for n in range(0,len(I2)):
    I2[n,:,:][I2[n,:,:]>0] = 1


I3 = I*I2
mean = []
for n in range(0,len(I3)):
    mean.append(np.mean(I3[n,:,:][I3[n,:,:]>0]))
    
plt.figure(figsize=(5,5))
plt.plot(mean)
plt.show()

