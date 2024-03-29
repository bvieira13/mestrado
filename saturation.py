# Importantod bibliotecas necessárias para o tratamento dos dados
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import imfun as im
import shutil

from scipy.signal import butter,filtfilt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class dir_t:
    root:str
    common:str

class model_t:
    type:str
    date:str
    specimen:str
    stage:str


class saturation_t:
# Construtor da classe para cálculo dos parâmetros de saturação
# em que são setados o diretório raiz, as pastas em comum e o 
# tipo de experimento que será desenvolvido
    def __init__(self, path:dir_t):
        self.dir = path.root + path.common + '\\'
# Carrega os valores de absorbância da oxi e desoxihemoglobina e 
# seus respectivos comprimentos de onda, ou seja a curva espectral
    def load_absorption(self):
        abs_data = np.loadtxt('hb_hb02_absorption.txt', usecols=range(0,3));
        wavelength = abs_data[:,0];
        hb02_absorption = abs_data[:,1];
        hb_absorption = abs_data[:,2];
        return wavelength, hb02_absorption, hb_absorption

# Carrega as imagens presentes no diretório definido pelo usuário
    def load_hsi(self,model:model_t):
        # Definindo o diretório onde as imagens estão alocadas
        dir = self.dir + (model.type + '\\' + model.date + '\\' + model.specimen + '\\' + model.stage)
        # Carregando os nomes dos imagens presentes no diretório
        self.img_names = os.listdir(dir)
        # Definindo um objeto do tipo lista para armazenar as imagens
        img = []
        # Carregando as imagens para a lista
        for name in self.img_names:
            file_dir = dir + '\\' + name
            temp = cv.imread(file_dir,-1)
            img.append(temp)
        # Convertendo a imagem para o tipo array
        self.hsi = np.asarray(img)

# Calcula a média dos pixels uma região retangular definida pelo usuário 
    def rectangle_roi_mean(self, img, index:np.uint8 = None):
        if index is None:
            [img_mask, points] = im.polyroi(img)

            dim_x = []
            dim_y = []

            for i in range(0, len(points)):
                (y,x) = points[i]
                dim_x.append(x)
                dim_y.append(y)
            
            xmin = np.min(dim_x)
            xmax = np.max(dim_x)

            ymin = np.min(dim_y)
            ymax = np.max(dim_y)

            crop_roi = img[xmin:xmax,ymin:ymax]
            mean = np.mean(crop_roi)
            std = np.std(crop_roi)
        else: 
            [img_mask, points] = im.polyroi(img[index])

            dim_x = []
            dim_y = []

            for i in range(0, len(points)):
                (y,x) = points[i]
                dim_x.append(x)
                dim_y.append(y)
            
            xmin = np.min(dim_x)
            xmax = np.max(dim_x)

            ymin = np.min(dim_y)
            ymax = np.max(dim_y)

            mean = []
            std = []
            for m in range(0, len(img)):
                crop_roi = img[m][xmin:xmax,ymin:ymax]
                temp = np.mean(crop_roi)
                std.append(np.std(crop_roi))
                mean.append(temp)

        return mean, std

# Normaliza os valores dos pixels das imagens pela média da 
# referência utilizada
    def normalize(self, mean):
        img_norm = []

        for i in range(0, len(self.hsi)):
            img_norm.append(self.hsi[i][:,:]/mean[i])
        
        self.hsi_norm = img_norm

# Extrai a informação dos comprimentos de onda do nome das imagens     
    def extract_wavelenght_from_name(self, tag:str):        
        wl = []
        for data in self.img_names:
            temp = data.replace('.bmp', '')
            temp = temp.replace(tag,'')
            wl.append(int(temp))
        self.wavelenght = wl
# Equalização de histograma
    def equalize_histogram(self, img):
        # Computando o valo máximo e mínimo dos niveis de cinza presentes na imagem
        max = np.max(img);
        min = np.min(img);
        # Computando os valores da nova imagem após a transformação
        img_norm = (img.astype(np.float32) - min)/(max-min);
        return img_norm
# Faz o cálculo do valor de saturação dos pixels com base na absorbância da oxi e 
# desoxiemoglobina e retorna a imagem, o range ideal e a média na região do tumor 
# de cada imagem
    def so2_map(self, points:tuple = []):
        wl, hb02_absorption, hb_absorption = self.load_absorption()
        wl_array = list(wl)  

        img_norm = self.hsi_norm
        wavelength = self.wavelenght
        
        if len(points) > 1:
            so2 = []
            for point in points:
                (isosbestic, non_isosbestic) = point
                numerator = (hb_absorption[wl_array.index(non_isosbestic)] - hb_absorption[wl_array.index(isosbestic)]*np.divide(np.log10(img_norm[wavelength.index(non_isosbestic)]),np.log10(img_norm[wavelength.index(isosbestic)])))
                denominator = (hb_absorption[wl_array.index(non_isosbestic)] - hb02_absorption[wl_array.index(non_isosbestic)])
                temp = np.abs(np.divide(numerator,denominator))*100
                temp[temp>100] = 100
                so2.append(temp)
            
            self.so2_norm = []
            for img in so2:
                temp = cv.normalize(img, None, 1, 0, cv.NORM_MINMAX, cv.CV_32F)
                self.so2_norm.append(temp)

        else:
            (isosbestic, non_isosbestic) = points[0]
            numerator = (hb_absorption[wl_array.index(non_isosbestic)] - hb_absorption[wl_array.index(isosbestic)]*np.divide(img_norm[wavelength.index(isosbestic)],img_norm[wavelength.index(non_isosbestic)]))
            denominator = (hb_absorption[wl_array.index(non_isosbestic)] - hb02_absorption[wl_array.index(non_isosbestic)])
            so2 = np.abs(np.divide(numerator,denominator))
            so2[so2>4.9] = 5
            temp = cv.normalize(so2, None, 1, 0, cv.NORM_MINMAX, cv.CV_32F)
            self.so2_norm = temp
                
        return so2

# Gera o mapa com o padrão de saturação de oxigenação a partir de 
# comprimentos de onda e intervalo de intensidade definidos 
    def plot_so2_map(self, img, range:tuple = None):
        fig = plt.figure(figsize=(8,8))
        if range != None:
            (min, max) = range
            plt.imshow(img, cmap = 'RdGy_r', vmin=min, vmax=max)
        else:
            plt.imshow(img, cmap = 'RdGy_r')
        plt.title(self.title)
        plt.colorbar(shrink=0.65)
        plt.savefig(self.filename)
        plt.close(fig)

    def replace_equal_file(self, folder:str):
        current_dir = os.getcwd();
        dst_folder = current_dir + '\\results\\' + folder + '\\'
        scr_folder = current_dir + '\\'
        if os.path.exists(dst_folder + self.filename):
            path = dst_folder + self.filename
            os.remove(path)
            shutil.move(scr_folder + self.filename, dst_folder + self.filename)
        else:
            shutil.move(scr_folder + self.filename, dst_folder + self.filename)

    def save_hand_fig(self, model:model_t, wavelength:tuple = []):
        range = [(0,875)]
        inf_const = '-' + model.specimen + '-' + model.stage
        mid_const = '-nonisos'
        N = 50
        self.load_hsi(model)
        self.extract_wavelenght_from_name(model.stage)
        hsi_mean, std = self.rectangle_roi_mean(self.hsi, self.wavelenght.index(530))
        self.normalize(hsi_mean)
        so2 = self.so2_map(wavelength)

        self.filename = 'isos' + str(wavelength[0][0]) + mid_const + str(wavelength[0][1])  + inf_const +'.png'
        # self.title = '$\lambda_1$ =' + str(wavelength[1]) + ' nm e $\lambda_2$ = ' + str(wavelength[0]) + ' nm'
        self.title = ''

        self.plot_so2_map(self.so2_norm,range[0])
        self.replace_equal_file('hand\\final')
        temp = cv.normalize(so2, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
        img, point = im.improfile(temp, cmap= cv.COLORMAP_HSV)
        perfil = (np.sum(img,axis=1) - 255)/255
        perfil[perfil>=1] = np.mean(perfil)
        x = np.convolve(perfil, np.ones(N)/N, mode='valid')
        perfil = 1 - x   
        so2_norm = temp

        vessel_mean, vessel_std = self.rectangle_roi_mean(so2_norm)
        vessel_mean /= 255
        vessel_std /= 255
        mean = 1 - vessel_mean
        std = vessel_std
        return so2_norm, perfil, mean, std
        


    def save_mice_fig(self, model:model_t, wavelength:tuple = []):
        dates = ['2022.09.28', '2022.09.29', '2022.10.03']
        day_index = ['1', '2', '6']

        sup_const = model.specimen + '-' + model.stage + '-' + 'isos'
        mid_const = '-nonisos'
        inf_const = '-day-'
        i = 0

        so2_norm = []
        perfil = []
        mean = []
        std = []
        N = 20
        for date in dates:
            model.date = date
            self.load_hsi(model)
            self.extract_wavelenght_from_name('cam_')
            hsi_mean, dummy = self.rectangle_roi_mean(self.hsi, self.wavelenght.index(530))
            self.normalize(hsi_mean)
            so2 = self.so2_map(wavelength)

            self.filename =  sup_const + str(wavelength[0][0]) + mid_const + str(wavelength[0][1]) + inf_const + day_index[i] + '.png'
            self.title = 'Dia ' + day_index[i] + ' ($\lambda_1$ =' + str(wavelength[0][1]) + ' nm e $\lambda_2$ = ' + str(wavelength[0][0]) + ' nm)'
            self.plot_so2_map(so2)
            self.replace_equal_file('mouse\\final')
            temp = cv.normalize(so2, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
            so2_norm.append(temp)
            img, point = im.improfile(temp, cmap= cv.COLORMAP_HSV)
            value = (np.sum(img,axis=1) - 255)/255
            value[value>1] = np.mean(value)    
            t_mean, t_std = self.rectangle_roi_mean(temp)
            x = np.convolve(value, np.ones(N)/N, mode='valid')
            perfil.append(x)
            mean.append(1 - t_mean/255)
            std.append(t_std/255)
            i+=1

        return so2_norm, perfil, mean, std



def main():
    path = dir_t()
    path.root = 'D:\\Documents\\Graduate\\Master\\Research\\Code\\'
    path.common = 'dataset\\Hemoglobin'
    
    data = saturation_t(path)
    model = model_t()

    model.type = 'Nude Mouse Melanoma'
    model.specimen = 'green-induced-04'
    model.stage = 'affect'

    # isos_wl = [(452,540), (584,562)]
    isos_wl = [(452,540)]

    g1_so2, g1_perfil, g1_mean, g1_std = data.save_mice_fig(model, isos_wl)
    g1_so2_c = np.subtract(255, g1_so2)
    
    # model.specimen = 'green-induced-02'

    # g2_so2, g2_perfil, g2_mean, g2_std = data.save_mice_fig(model, isos_wl)
    # g2_so2_c = np.subtract(255, g2_so2)

    # model.specimen = 'green-induced-03'

    # g3_so2, g3_perfil, g3_mean, g3_std = data.save_mice_fig(model, isos_wl)
    # g3_so2_c = np.subtract(255, g3_so2)
    
    # model.specimen = 'green-induced-04'

    # g4_so2, g4_perfil, g4_mean, g4_std = data.save_mice_fig(model, isos_wl)
    # g4_so2_c = np.subtract(255, g4_so2)


    # plt.rcParams['font.size'] = 12
    # vmin_oxi0 = vmin_oxi1 = vmin_oxi2 = 0.0
    # vmax_oxi0 = vmax_oxi1 = vmax_oxi2 = 0.5
    # vmin_deox0 = vmin_deox1 = vmin_deox2 = 0.5
    # vmax_deox0 = vmax_deox1 = vmax_deox2 = 1.00
    
    # fig, ax = plt.subplots(2,3)
    # fig.set_figwidth(10)
    # fig.set_figheight(8)
    # img = ax[0,0].imshow((g1_so2[0]-np.min(g1_so2[0]))/np.max(g1_so2[0]-np.min(g1_so2[0])),vmin=vmin_oxi0,vmax=vmax_oxi0,cmap = 'RdGy_r')
    # ax[0,0].axis('off')
    # divider = make_axes_locatable(ax[0,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,0].imshow((g1_so2_c[0]-np.min(g1_so2_c[0]))/np.max(g1_so2_c[0]-np.min(g1_so2_c[0])),vmin=vmin_deox0,vmax=vmax_deox0,cmap = 'RdGy_r')
    # ax[1,0].axis('off')
    # divider = make_axes_locatable(ax[1,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,1].imshow((g1_so2[1]-np.min(g1_so2[0]))/np.max(g1_so2[0]-np.min(g1_so2[0])),vmin=vmin_oxi1,vmax=vmax_oxi1,cmap = 'RdGy_r')
    # ax[0,1].axis('off')
    # divider = make_axes_locatable(ax[0,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,1].imshow((g1_so2_c[1]-np.min(g1_so2_c[0]))/np.max(g1_so2_c[0]-np.min(g1_so2_c[0])),vmin=vmin_deox1,vmax=vmax_deox1,cmap = 'RdGy_r')
    # ax[1,1].axis('off')
    # divider = make_axes_locatable(ax[1,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,2].imshow((g1_so2[2]-np.min(g1_so2[2]))/np.max(g1_so2[2]-np.min(g1_so2[2])),vmin=vmin_oxi2,vmax=vmax_oxi2,cmap = 'RdGy_r')
    # ax[0,2].axis('off')
    # divider = make_axes_locatable(ax[0,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,2].imshow((g1_so2_c[2]-np.min(g1_so2_c[2]))/np.max(g1_so2_c[2]-np.min(g1_so2_c[2])),vmin=vmin_deox2,vmax=vmax_deox2,cmap = 'RdGy_r')
    # ax[1,2].axis('off')
    # divider = make_axes_locatable(ax[1,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # plt.tight_layout()
    # plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(2,3)
    # fig.set_figwidth(10)
    # fig.set_figheight(8)
    # img = ax[0,0].imshow((g2_so2[0]-np.min(g2_so2[0]))/np.max(g2_so2[0]-np.min(g2_so2[0])),vmin=vmin_oxi0,vmax=vmax_oxi0,cmap = 'RdGy_r')
    # ax[0,0].axis('off')
    # divider = make_axes_locatable(ax[0,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,0].imshow((g2_so2_c[0]-np.min(g2_so2_c[0]))/np.max(g2_so2_c[0]-np.min(g2_so2_c[0])),vmin=vmin_deox0,vmax=vmax_deox0,cmap = 'RdGy_r')
    # ax[1,0].axis('off')
    # divider = make_axes_locatable(ax[1,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,1].imshow((g2_so2[1]-np.min(g2_so2[1]))/np.max(g2_so2[1]-np.min(g2_so2[1])),vmin=vmin_oxi1,vmax=vmax_oxi1,cmap = 'RdGy_r')
    # ax[0,1].axis('off')
    # divider = make_axes_locatable(ax[0,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,1].imshow((g2_so2_c[1]-np.min(g2_so2_c[1]))/np.max(g2_so2_c[1]-np.min(g2_so2_c[1])),vmin=vmin_deox1,vmax=vmax_deox1,cmap = 'RdGy_r')
    # ax[1,1].axis('off')
    # divider = make_axes_locatable(ax[1,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,2].imshow((g2_so2[2]-np.min(g2_so2[2]))/np.max(g2_so2[2]-np.min(g2_so2[2])),vmin=vmin_oxi2,vmax=vmax_oxi2,cmap = 'RdGy_r')
    # ax[0,2].axis('off')
    # divider = make_axes_locatable(ax[0,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,2].imshow((g2_so2_c[2]-np.min(g2_so2_c[2]))/np.max(g2_so2_c[2]-np.min(g2_so2_c[2])),vmin=vmin_deox2,vmax=vmax_deox2,cmap = 'RdGy_r')
    # ax[1,2].axis('off')
    # divider = make_axes_locatable(ax[1,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # plt.tight_layout()
    # plt.tight_layout()
    # plt.show()
    
    # fig, ax = plt.subplots(2,3)
    # fig.set_figwidth(10)
    # fig.set_figheight(8)
    # img = ax[0,0].imshow((g3_so2[0]-np.min(g3_so2[0]))/np.max(g3_so2[0]-np.min(g3_so2[0])),vmin=vmin_oxi0,vmax=vmax_oxi0,cmap = 'RdGy_r')
    # ax[0,0].axis('off')
    # divider = make_axes_locatable(ax[0,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,0].imshow((g3_so2_c[0]-np.min(g3_so2_c[0]))/np.max(g3_so2_c[0]-np.min(g3_so2_c[0])),vmin=vmin_deox0,vmax=vmax_deox0,cmap = 'RdGy_r')
    # ax[1,0].axis('off')
    # divider = make_axes_locatable(ax[1,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,1].imshow((g3_so2[1]-np.min(g3_so2[1]))/np.max(g3_so2[1]-np.min(g3_so2[1])),vmin=vmin_oxi1,vmax=vmax_oxi1,cmap = 'RdGy_r')
    # ax[0,1].axis('off')
    # divider = make_axes_locatable(ax[0,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,1].imshow((g3_so2_c[1]-np.min(g3_so2_c[1]))/np.max(g3_so2_c[1]-np.min(g3_so2_c[1])),vmin=vmin_deox1,vmax=vmax_deox1,cmap = 'RdGy_r')
    # ax[1,1].axis('off')
    # divider = make_axes_locatable(ax[1,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,2].imshow((g3_so2[2]-np.min(g3_so2[2]))/np.max(g3_so2[2]-np.min(g3_so2[2])),vmin=vmin_oxi2,vmax=vmax_oxi2,cmap = 'RdGy_r')
    # ax[0,2].axis('off')
    # divider = make_axes_locatable(ax[0,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,2].imshow((g3_so2_c[2]-np.min(g3_so2_c[2]))/np.max(g3_so2_c[2]-np.min(g3_so2_c[2])),vmin=vmin_deox2,vmax=vmax_deox2,cmap = 'RdGy_r')
    # ax[1,2].axis('off')
    # divider = make_axes_locatable(ax[1,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # plt.tight_layout()
    # plt.tight_layout()
    # plt.show()
    
    # fig, ax = plt.subplots(2,3)
    # fig.set_figwidth(10)
    # fig.set_figheight(8)
    # img = ax[0,0].imshow((g4_so2[0]-np.min(g4_so2[0]))/np.max(g4_so2[0]-np.min(g4_so2[0])),vmin=vmin_oxi0,vmax=vmax_oxi0,cmap = 'RdGy_r')
    # ax[0,0].axis('off')
    # divider = make_axes_locatable(ax[0,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,0].imshow((g4_so2_c[0]-np.min(g4_so2_c[0]))/np.max(g4_so2_c[0]-np.min(g4_so2_c[0])),vmin=vmin_deox0,vmax=vmax_deox0,cmap = 'RdGy_r')
    # ax[1,0].axis('off')
    # divider = make_axes_locatable(ax[1,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,1].imshow((g4_so2[1]-np.min(g4_so2[1]))/np.max(g4_so2[1]-np.min(g4_so2[1])),vmin=vmin_oxi1,vmax=vmax_oxi1,cmap = 'RdGy_r')
    # ax[0,1].axis('off')
    # divider = make_axes_locatable(ax[0,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,1].imshow((g4_so2_c[1]-np.min(g4_so2_c[1]))/np.max(g4_so2_c[1]-np.min(g4_so2_c[1])),vmin=vmin_deox1,vmax=vmax_deox1,cmap = 'RdGy_r')
    # ax[1,1].axis('off')
    # divider = make_axes_locatable(ax[1,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,2].imshow((g4_so2[2]-np.min(g4_so2[2]))/np.max(g4_so2[2]-np.min(g4_so2[2])),vmin=vmin_oxi2,vmax=vmax_oxi2,cmap = 'RdGy_r')
    # ax[0,2].axis('off')
    # divider = make_axes_locatable(ax[0,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,2].imshow((g4_so2_c[2]-np.min(g4_so2_c[2]))/np.max(g4_so2_c[2]-np.min(g4_so2_c[2])),vmin=vmin_deox2,vmax=vmax_deox2,cmap = 'RdGy_r')
    # ax[1,2].axis('off')
    # divider = make_axes_locatable(ax[1,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # plt.tight_layout()
    # plt.tight_layout()
    # plt.show()
    
    # x_data = [1, 2, 6]
    # x_fit = np.linspace(0,10,1000)

    # coeff = np.polyfit(x_data,g1_mean, 2)
    # y_g1 = np.poly1d(coeff)

    # coeff = np.polyfit(x_data,g2_mean, 2)
    # y_g2 = np.poly1d(coeff)

    # coeff = np.polyfit(x_data,g3_mean, 2)
    # y_g3 = np.poly1d(coeff)

    # coeff = np.polyfit(x_data,g4_mean, 2)
    # y_g4 = np.poly1d(coeff)

    # plt.rcParams['font.size'] = 18

    # plt.figure(figsize=(8,8))
    # plt.errorbar(x_data, g1_mean, yerr = g1_std,  fmt='o', color= 'black', 
    #             ecolor = 'black', elinewidth = 1, capsize = 2, label='Espécime 1')
    # plt.plot(x_fit,y_g1(x_fit),'--k')

    # plt.errorbar(x_data, g2_mean, yerr = g2_std,  fmt='o', color= 'blue', 
    #             ecolor = 'blue', elinewidth = 1, capsize = 2, label='Espécime 2')
    # plt.plot(x_fit,y_g2(x_fit),'--b')
    # plt.errorbar(x_data, g3_mean, yerr = g3_std,  fmt='o', color= 'red', 
    #             ecolor = 'red', elinewidth = 1, capsize = 2, label='Espécime 3')
    # plt.plot(x_fit,y_g3(x_fit),'--r')

    # plt.errorbar(x_data, g4_mean, yerr = g4_std,  fmt='o', color= 'green', 
    #             ecolor = 'green', elinewidth = 1, capsize = 2, label='Espécime 4')
    # plt.plot(x_fit,y_g4(x_fit),'--g')
    # plt.axis([0, 10, 0, 1.2])
    # plt.ylabel("Saturação de O$_2$ média normalizada")
    # plt.xlabel("Tempo [dias]")
    # plt.legend()
    # plt.show()
    len_ = np.max([len(g1_perfil[0]),len(g1_perfil[1]),len(g1_perfil[2])])
    var = (len_ % 50) - 50
    x_f = len_ + var
    plt.figure(figsize=(8,8))
    plt.plot(g1_perfil[0], 'g', label='Dia 1')
    plt.plot(g1_perfil[1], 'y', label='Dia 2')
    plt.plot(g1_perfil[2], 'b', label='Dia 5')
    plt.legend()
    plt.axis([0, int(x_f), 0, 1])
    plt.ylabel("Saturação de O$_2$ [normalizada]")
    plt.xlabel("Posição [distância entre pixels]")
    plt.show()

    # plt.figure(figsize=(8,8))
    # plt.plot(g2_perfil[0], 'k', label='Dia 1')
    # plt.plot(g2_perfil[1], 'b', label='Dia 2')
    # plt.plot(g2_perfil[2], 'r', label='Dia 5')
    # plt.legend()
    # plt.axis([0, 600, 0, 1.4])
    # plt.ylabel("Saturação de O$_2$ [normalizada]")
    # plt.xlabel("Posição [distância entre pixels]")
    # plt.show()

    # plt.figure(figsize=(8,8))
    # plt.plot(g4_perfil[0], 'k', label='Dia 1')
    # plt.plot(g4_perfil[1], 'b', label='Dia 2')
    # plt.plot(g4_perfil[2], 'r', label='Dia 5')
    # plt.legend()
    # plt.axis([0, 600, 0, 1.4])
    # plt.ylabel("Saturação de O$_2$ [normalizada]")
    # plt.xlabel("Posição [distância entre pixels]")
    # plt.show()

    # model.stage = 'healthy'
    # for wl in isos_wl:
    #     data.save_mice_fig(model, wl)

    # model.type = 'Healthy Human Hand'
    # model.specimen = 'Marlon'
    # model.date = '2022.12.14'

    # point = [(500,710)]

    # model.stage = 'repouso'
    # r_so2, r_perfil, r_mean, r_std = data.save_hand_fig(model,point)
    # r_so2_c = 255 - r_so2
    # model.stage = 'oclusao'
    # o_so2, o_perfil, o_mean, o_std = data.save_hand_fig(model,point)
    # o_so2_c = 255 - o_so2
    # model.stage = 'liberacao'
    # l_so2, l_perfil, l_mean, l_std = data.save_hand_fig(model,point)
    # l_so2_c = 255 - l_so2

    
    # x_data = ['Repouso', 'Oclusão', 'Liberação']
    # y_data = [r_mean, o_mean, l_mean]
    # y_data_err = [r_std, o_std, l_std]
    
    # print(y_data)
    # print(y_data_err)
    # plt.rcParams['font.size'] = 18
    # plt.figure(figsize=(8,8))
    # plt.bar(x_data, y_data, yerr = y_data_err, alpha = 0.5, 
    #         color= 'blue', ecolor = 'black', capsize = 2)
    # plt.axis([-1, 3, 0, 0.5])
    # plt.grid(True)            
    # plt.ylabel("Saturação de O$_2$ média normalizada")
    # plt.show()

    # plt.figure(figsize=(8,8))
    # plt.plot(r_perfil, 'k', label='Repouso')
    # plt.plot(o_perfil, 'r', label='Oclusão')
    # plt.plot(l_perfil, 'b', label='Liberação')
    # plt.legend()
    # plt.axis([0, 300, 0, 1])
    # plt.ylabel("Saturação de O$_2$ [normalizada]")
    # plt.xlabel("Posição [distância entre pixels]")
    # plt.show()

    # plt.rcParams['font.size'] = 12
    # vmin_oxi0 = vmin_oxi1 = vmin_oxi2 = 0.0
    # vmax_oxi0 = vmax_oxi1 = vmax_oxi2 = 0.5
    # vmin_deox0 = vmin_deox1 = vmin_deox2 = 0.5
    # vmax_deox0 = vmax_deox1 = vmax_deox2 = 1.00
    
    # fig, ax = plt.subplots(2,3)
    # fig.set_figwidth(10)
    # fig.set_figheight(8)
    # img = ax[0,0].imshow((r_so2-np.min(r_so2))/np.max(r_so2-np.min(r_so2)),vmin=vmin_oxi0,vmax=vmax_oxi0,cmap = 'RdGy_r')
    # ax[0,0].axis('off')
    # divider = make_axes_locatable(ax[0,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,0].imshow((r_so2_c-np.min(r_so2_c))/np.max(r_so2_c-np.min(r_so2_c)),vmin=vmin_deox0,vmax=vmax_deox0,cmap = 'RdGy_r')
    # ax[1,0].axis('off')
    # divider = make_axes_locatable(ax[1,0])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,1].imshow((o_so2-np.min(o_so2))/np.max(o_so2-np.min(o_so2)),vmin=vmin_oxi1,vmax=vmax_oxi1,cmap = 'RdGy_r')
    # ax[0,1].axis('off')
    # divider = make_axes_locatable(ax[0,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,1].imshow((o_so2_c-np.min(o_so2_c))/np.max(o_so2_c-np.min(o_so2_c)),vmin=vmin_deox1,vmax=vmax_deox1,cmap = 'RdGy_r')
    # ax[1,1].axis('off')
    # divider = make_axes_locatable(ax[1,1])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[0,2].imshow((l_so2-np.min(l_so2))/np.max(l_so2-np.min(l_so2)),vmin=vmin_oxi2,vmax=vmax_oxi2,cmap = 'RdGy_r')
    # ax[0,2].axis('off')
    # divider = make_axes_locatable(ax[0,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # img = ax[1,2].imshow((l_so2_c-np.min(l_so2_c))/np.max(l_so2_c-np.min(l_so2_c)),vmin=vmin_deox2,vmax=vmax_deox2,cmap = 'RdGy_r')
    # ax[1,2].axis('off')
    # divider = make_axes_locatable(ax[1,2])
    # ax_cb = divider.append_axes('right', size='5%', pad=0.1)
    # plt.colorbar(img, cax=ax_cb)
    # plt.tight_layout()
    # plt.tight_layout()
    
    plt.show()
main()