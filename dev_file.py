# %% 
# Importantod bibliotecas necessárias para o tratamento dos dados
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import imfun as im
# %%
# Definição de funções genéricas
def show_img(img, cmap='gray'):
    plt.figure(figsize=(10,8))
    plt.imshow(img, cmap)
    plt.show()

def equalize_histogram(img):
  # Computando o valo máximo e mínimo dos niveis de cinza presentes na imagem
  max = np.max(img)
  min = np.min(img)
  # Computando os valores da nova imagem após a transformação
  img_norm = (img.astype(np.float32) - min)*255/(max-min)
  img_out = img_norm.astype(np.uint8)
  return img_out

def load_absorption():
    abs_data = np.loadtxt('hb_hb02_absorption.txt', usecols=range(0,3));
    wavelength = abs_data[:,0];
    hb02_absorption = abs_data[:,1];
    hb_absorption = abs_data[:,2];
    return wavelength, hb02_absorption, hb_absorption

# %% 
# Varáveis de definição do diretório e pasta dos quais serão extraidos os dados
# root_path = 'C:\\Users\\Bruno Vieira\\Documents\\Mestrado\\'
root_path = 'D:\\Documents\\Graduate\\Master\\Research\\Results\\'
dir = 'dataset\\Hemoglobina\\Camundongo Melanoma Erika\\2022.10.03 - Camundongo\\'
folder = 'CVerd01\\FA'

data_path = root_path + dir + folder 
# Carregando os nomes dos arquivos presentes na pasta selecionada
names = os.listdir(data_path)
names.reverse()
# %%
# Definição de um objeto do tipo lista para armazenar as imagens presentes nos diretórios
img = []
# Carregando as imagens no objeto lista
for name in names:
    file_dir = data_path + '\\' + name 
    temp = cv.imread(file_dir, -1)
    img.append(temp)
# Convertendo a imagem para o tipo array
img = np.asarray(img)

# %%
# Selecionando regiões de interesse e armazenando-as a uma variável
img_crop, points = im.crop_poly_multiple(img)
# Convertendo a imagem da região de interesse para o tipo array
img_crop = np.asarray(img_crop)
# %%
# Convertendo os valores dos pixels de uint8_t para binário
for n in range(0,len(img_crop)):
    img_crop[n,:,:][img_crop[n,:,:]>0] = 1
# Obtendo a média dos valores dos pixels da região de interesse
img_roi = img_crop*img
# Calculando as médias das regiões de interesse 
mean = []
for n in range(0,len(img_roi)):
    mean.append(np.mean(img_roi[n,:,:][img_roi[n,:,:]>0]))
# %%
# Normalizando as imagens de cada comprimento de onda
img_norm = []

for i in range(0, len(img)):
    img_norm.append(img[i][:,:]/mean[i])

# %%
# Mostrando o mapa de calor da oxigenação
wl, hb02_absorption, hb_absorption = load_absorption()
wavelength = list([586, 584, 576, 570, 562, 556, 546, 540, 
                   530, 506, 500, 480, 452, 432, 422, 414])

wl_array = list(wl)

(isosbestic, non_isosbestic) = (570, 586)

so2 = (hb_absorption[wl_array.index(isosbestic)] - hb_absorption[wl_array.index(non_isosbestic)]*np.divide(img_norm[wavelength.index(non_isosbestic)], 
            img_norm[wavelength.index(isosbestic)]))/(hb_absorption[wl_array.index(isosbestic)] - hb02_absorption[wl_array.index(isosbestic)]);

plt.figure(figsize=(5,5));
plt.imshow(so2, cmap= 'RdGy_r', vmin = 0, vmax= 25);
plt.colorbar();
plt.show()
