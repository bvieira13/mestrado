import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import imfun
import shutil


def region_mean(dir):
    names = os.listdir(dir)
    names.reverse();

    I=[]

    for name in names:
        I.append(cv.imread(dir+'\\'+name, -1))

    I = np.asarray(I)

    I2, points = imfun.crop_poly_multiple(I)
    I2 = np.asarray(I2)

    for n in range(0,len(I2)):
        I2[n,:,:][I2[n,:,:]>0] = 1


    I3 = I*I2
    mean = []
    for n in range(0,len(I3)):
        mean.append(np.mean(I3[n,:,:][I3[n,:,:]>0]))
    
    return mean

current_dir = os.getcwd() 
data_path = current_dir + '\\dataset\\Hemoglobin\\Caracterization\\2022.09.22\\position\\'
dst_folder = current_dir + '\\results\\'
scr_folder = current_dir + '\\'

file_name = 'light_source_caracterization.png'


wavelength = ['586', '584', '576', '570', '562', '556', '546', '540', 
              '530', '506', '500', '480', '452', '432', '422', '414']
position = [-2, -1, 0, 1, 2, 3];
mean = []

for i in range(1, 7):
    dir = data_path + '0' + str(i)
    vec = region_mean(dir)
    vec.reverse()
    mean.append(vec)
    
mean = np.array(mean)
wavelength.reverse()
t_mean = np.transpose(mean)
norm_mean = np.divide(t_mean, 255)
count = 0
# usar um comando if para ter dois modelos de linha no plot e assim diferenciar
plt.figure(figsize=(10,8))
for data in norm_mean:
    plt.plot(position, data, label=wavelength[count] + " nm")
    count += 1

plt.ylabel("Intensidade luminosa normalizada")
plt.xlabel("Posição [cm]")
plt.axis([-2, 4, 0, 1]);

plt.legend()
plt.savefig(file_name)
plt.show()

if os.path.exists(dst_folder + file_name):
    path = dst_folder + file_name
    os.remove(path)
    shutil.move(scr_folder + file_name, dst_folder + file_name)
else:
    shutil.move(scr_folder + file_name, dst_folder + file_name)





