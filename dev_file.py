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
# %% 
# Varáveis de definição do diretório e pasta dos quais serão extraidos os dados
root_path = 'D:\\Documents\\Graduate\\Master\\Research\\'
dir = 'Results\\dataset\\Hemoglobina\\Camundongo Melanoma Erika\\2022.09.30 - Camundongo\\'
folder = 'CAzul01\\FA'

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
print(mean)