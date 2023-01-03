import cv2
import matplotlib.pyplot as plt
# dir_imfun = 'C:\\Users\\marlo\\My Drive\\College\\Biophotonics Lab\\Research\\Programs\Python\\Camera & Image'
import os
# os.chdir(dir_imfun)
import imfun
import tqdm
import numpy as np
from scipy.optimize import least_squares
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Loading images/image dir
root = os.getcwd() + '\\dataset\\Hemoglobin'
# dir0 = 'G:\\Shared drives\\Imageamento Multiespectral\\Imagens\\Hemoglobina\\2022.12.14 - Mao Voluntario Saudavel - Marlon\\repouso'
# dir1 = 'G:\\Shared drives\\Imageamento Multiespectral\\Imagens\\Hemoglobina\\2022.12.14 - Mao Voluntario Saudavel - Marlon\\oclusao'
# dir2 = 'G:\\Shared drives\\Imageamento Multiespectral\\Imagens\\Hemoglobina\\2022.12.14 - Mao Voluntario Saudavel - Marlon\\liberacao'

# dir0 = root  + '\\Nude Mouse Melanoma\\2022.09.28\\green-induced-01\\affect'
# dir1 = root  + '\\Nude Mouse Melanoma\\2022.09.29\\green-induced-01\\affect'
# dir2 = root  + '\\Nude Mouse Melanoma\\2022.10.03\\green-induced-01\\affect'

dir0 = root  + '\\Healthy Human Hand\\2022.12.14\\Marlon\\repouso'
dir1 = root  + '\\Healthy Human Hand\\2022.12.14\\Marlon\\oclusao'
dir2 = root  + '\\Healthy Human Hand\\2022.12.14\\Marlon\\liberacao'

def obtain_names(directory):
    names = []
    wavelengths = []
    for name in os.listdir(directory):
        names.append(os.path.join(directory, name))
        splited = [int(s) for s in [*name] if s.isdigit()]
        if len(splited) == 3:
            wavelength = int(splited[2]+splited[1]*10+splited[0]*100)
        wavelengths.append(wavelength)
    
    return names, wavelengths

names0, wavelengths = obtain_names(dir0)
names1, wavelengths = obtain_names(dir1)
names2, wavelengths = obtain_names(dir2)

# The 'HbO2' and 'Hb' exctinction coefficients are in cm-1/M
HbO2 = np.array([314, 276, 319.6, 610, 2128, 26600.4, 34639.6, 55540, 44496,
                 32620, 34476.8, 49868, 53236, 39956.8, 19946, 20932.8,
                 26629.2, 58864, 214120, 431880, 524280, 266232])
Hb = np.array([1540.48, 2051.96, 3226.56, 5148.8, 12567.6, 32851.6, 34332.8,
               40092, 45072, 52276, 54540, 51268, 46592, 39036.4, 23774.4,
               20862, 14550, 62640, 552160, 429880, 342596, 223296])

def hemoglobin_map(names, wavelengths, HbO2, Hb, mask, resize, **kwargs):
    start = kwargs.get('start')
    stop = kwargs.get('stop')
    maximum = 100
    minimum = 0
    if not start: start = 0
    if not stop: stop = 0 
    images = []
    for name in names:
        image = cv2.imread(name)[:,:,0]
        images.append(cv2.resize(image,(int(len(image[0,:])/resize),
                                        int(len(image[:,0])/resize)),
                                 cv2.INTER_CUBIC))
    
    image_map_HbO2 = np.zeros(np.shape(images[0]))
    image_map_Hb = np.zeros(np.shape(images[0]))
    loop = tqdm.tqdm(range(np.shape(images[0])[0]))
    cost= []
    for l in loop:
        for c in range(np.shape(images[0])[1]):
            absorption = []
            HbO2vec = []
            Hbvec = []
            for n in range(start, len(names)-stop):
                try: absorption.append(-np.log(images[n][l,c]/np.mean(images[n][mask>0])))
                except: absorption.append(int(0))
                HbO2vec.append(HbO2[n])
                Hbvec.append(Hb[n])
            absorption = np.asarray(absorption)
            HbO2vec = np.asarray(HbO2vec)
            Hbvec = np.array(Hbvec)
            
            def func_to_minimize(x):
                return np.array((HbO2vec*x[0]+Hbvec*(1-x[0]))*x[1]+x[2]-absorption)
            
            try:
                result = least_squares(func_to_minimize, [0,0,0])
                HbO2con = result.x[0]
                cost.append(result.cost)
            except: HbO2con = 0
            if (HbO2con > maximum): HbO2con = maximum
            elif (HbO2con<minimum): HbO2con = minimum
            image_map_HbO2[l,c] = HbO2con
            image_map_Hb[l,c] = 1-HbO2con
    return image_map_HbO2, image_map_Hb, cost

resize = 3
image = cv2.imread(names0[2])[:,:,0]
image_temp = cv2.resize(image,(int(len(image[0,:])/resize),
                               int(len(image[:,0])/resize)), cv2.INTER_CUBIC)
mask, points = imfun.polyroi(image_temp, 
                             window_name='Choose a region for normalization')    
image_map_HbO2_0, image_map_Hb_0, cost0 = hemoglobin_map(names0, wavelengths, HbO2,
                                                  Hb, mask, resize=resize, start=5, stop=4)
print('\n\ncost:', np.mean(cost0))

temp = cv2.normalize(image_map_HbO2_0, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
img, point = imfun.improfile(temp, cmap= cv2.COLORMAP_HSV)
perfil = (np.sum(img,axis=1) - 255)/255
perfil[perfil>1] = 1
r_perfil = perfil
plt.figure(figsize=(8,8))
plt.plot(r_perfil, 'k', label='Repouso')
plt.legend()
plt.axis([0, len(r_perfil), 0, 1.4])
plt.ylabel("Saturação de O$_2$ normalizada")
plt.xticks([])
plt.show()
prop, imask = imfun.imroiprop(temp)
r_mean, r_std = (prop[1]/255, prop[2]/255)
# image = cv2.imread(names1[4])[:,:,0]
# image_temp = cv2.resize(image,(int(len(image[0,:])/resize),
#                                int(len(image[:,0])/resize)), cv2.INTER_CUBIC)
# mask, points = imfun.polyroi(image_temp, 
#                              window_name='Choose a region for normalization')
image_map_HbO2_1, image_map_Hb_1, cost1 = hemoglobin_map(names1, wavelengths, HbO2,
                                                  Hb, mask, resize=resize, start=5, stop=4)
print('\n\ncost:', np.mean(cost1))
temp = cv2.normalize(image_map_HbO2_1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
img, point = imfun.improfile(temp, cmap= cv2.COLORMAP_HSV)
perfil = (np.sum(img,axis=1) - 255)/255
perfil[perfil>1] = 1
o_perfil = perfil
prop, imask = imfun.imroiprop(temp)
o_mean, o_std = (prop[1]/255, prop[2]/255)
# image = cv2.imread(names2[4])[:,:,0]
# image_temp = cv2.resize(image,(int(len(image[0,:])/resize),
#                                int(len(image[:,0])/resize)), cv2.INTER_CUBIC)
# mask, points = imfun.polyroi(image_temp, 
#                              window_name='Choose a region for normalization')
image_map_HbO2_2, image_map_Hb_2, cost2 = hemoglobin_map(names2, wavelengths, HbO2,
                                                  Hb, mask, resize=resize, start=5, stop=4)
print('\n\ncost:', np.mean(cost2))
temp = cv2.normalize(image_map_HbO2_2, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
img, point = imfun.improfile(temp, cmap= cv2.COLORMAP_HSV)
perfil = (np.sum(img,axis=1) - 255)/255
perfil[perfil>1] = 1
l_perfil = perfil
prop, imask = imfun.imroiprop(temp)
l_mean, l_std = (prop[1]/255, prop[2]/255)

x_data = ['Repouso', 'Oclusão', 'Liberação']
y_data = [r_mean, o_mean, l_mean]
y_data_err = [r_std, o_std, l_std]
plt.rcParams['font.size'] = 18
plt.figure(figsize=(8,8))
plt.bar(x_data, y_data, yerr = y_data_err, alpha = 0.5, 
        color= 'blue', ecolor = 'black', capsize = 2)
plt.axis([-1, 3, 0, 0.5])
plt.grid(True)            
plt.ylabel("Saturação de O$_2$ média normalizada")
plt.show()

plt.figure(figsize=(8,8))
plt.plot(r_perfil, 'k', label='Repouso')
plt.plot(o_perfil, 'r', label='Oclusão')
plt.plot(l_perfil, 'b', label='Liberação')
plt.legend()
plt.axis([0, len(r_perfil), 0, 1.4])
plt.ylabel("Saturação de O$_2$ normalizada")
plt.xticks([])
plt.show()
#%%
vmin_oxi0 = vmin_oxi1 = vmin_oxi2 = 0.00
vmax_oxi0 = vmax_oxi1 = vmax_oxi2 = 0.08
vmin_deox0 = vmin_deox1 = vmin_deox2 = 0.92
vmax_deox0 = vmax_deox1 = vmax_deox2 = 1.00

# vmin_oxi0 = 0.00
# vmax_oxi0 = 0.08
# vmin_deox0 = 0.92
# vmax_deox0 = 1.00
# vmin_oxi1 = 0.000
# vmax_oxi1 = 0.08
# vmin_deox1 = 0.92
# vmax_deox1 = 1.000
# vmin_oxi2 = 0.00
# vmax_oxi2 = 0.08
# vmin_deox2 = 0.92
# vmax_deox2 = 1.00
fig, ax = plt.subplots(2,3)
im = ax[0,0].imshow((image_map_HbO2_0-np.min(image_map_HbO2_0))/np.max(image_map_HbO2_0-np.min(image_map_HbO2_0)),vmin=vmin_oxi0,vmax=vmax_oxi0)
ax[0,0].axis('off')
divider = make_axes_locatable(ax[0,0])
ax_cb = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=ax_cb)
im = ax[1,0].imshow((image_map_Hb_0-np.min(image_map_Hb_0))/np.max(image_map_Hb_0-np.min(image_map_Hb_0)),vmin=vmin_deox0,vmax=vmax_deox0)
ax[1,0].axis('off')
divider = make_axes_locatable(ax[1,0])
ax_cb = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=ax_cb)
im = ax[0,1].imshow((image_map_HbO2_1-np.min(image_map_HbO2_1))/np.max(image_map_HbO2_1-np.min(image_map_HbO2_1)),vmin=vmin_oxi1,vmax=vmax_oxi1)
ax[0,1].axis('off')
divider = make_axes_locatable(ax[0,1])
ax_cb = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=ax_cb)
im = ax[1,1].imshow((image_map_Hb_1-np.min(image_map_Hb_1))/np.max(image_map_Hb_1-np.min(image_map_Hb_1)),vmin=vmin_deox1,vmax=vmax_deox1)
ax[1,1].axis('off')
divider = make_axes_locatable(ax[1,1])
ax_cb = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=ax_cb)
im = ax[0,2].imshow((image_map_HbO2_2-np.min(image_map_HbO2_2))/np.max(image_map_HbO2_2-np.min(image_map_HbO2_2)),vmin=vmin_oxi2,vmax=vmax_oxi2, cmap = 'RdGy_r')
ax[0,2].axis('off')
divider = make_axes_locatable(ax[0,2])
ax_cb = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=ax_cb)
im = ax[1,2].imshow((image_map_Hb_2-np.min(image_map_Hb_2))/np.max(image_map_Hb_2-np.min(image_map_Hb_2)),vmin=vmin_deox2,vmax=vmax_deox2, cmap = 'RdGy_r')
ax[1,2].axis('off')
divider = make_axes_locatable(ax[1,2])
ax_cb = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=ax_cb)
plt.tight_layout()
plt.tight_layout()
plt.show()
# fig, ax = plt.subplots(2,3)
# ax[0,0].imshow((image_map_HbO2_0-np.min(image_map_HbO2_0))/np.max(image_map_HbO2_0-np.min(image_map_HbO2_0)))
# ax[0,0].axis('off')
# ax[1,0].imshow((image_map_Hb_0-np.min(image_map_Hb_0))/np.max(image_map_Hb_0-np.min(image_map_Hb_0)))
# ax[1,0].axis('off')
# ax[0,1].imshow((image_map_HbO2_1-np.min(image_map_HbO2_1))/np.max(image_map_HbO2_1-np.min(image_map_HbO2_1)))
# ax[0,1].axis('off')
# ax[1,1].imshow((image_map_Hb_1-np.min(image_map_Hb_1))/np.max(image_map_Hb_1-np.min(image_map_Hb_1)))
# ax[1,1].axis('off')
# ax[0,2].imshow((image_map_HbO2_2-np.min(image_map_HbO2_2))/np.max(image_map_HbO2_2-np.min(image_map_HbO2_2)))
# ax[0,2].axis('off')
# ax[1,2].imshow((image_map_Hb_2-np.min(image_map_Hb_2))/np.max(image_map_Hb_2-np.min(image_map_Hb_2)))
# ax[1,2].axis('off')
# plt.tight_layout()
# plt.tight_layout()