import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import imfun
import shutil

def square_roi_statistic(hsi_img, wl_array:list, element):
    
    [img_mask, points] = imfun.polyroi(hsi_img[wl_array.index(element)])

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
    stdev = []

    for m in range(0, len(hsi_img)):
        crop_roi = hsi_img[m][xmin:xmax,ymin:ymax]
        temp = np.mean(crop_roi)
        mean.append(temp)
        stdev.append(np.std(crop_roi))

    return mean, stdev

def region_mean(dir:str, path):
    names = os.listdir(dir)
    
    I=[]
    wavelenght = extract_wavelenght_from_name(dir, path)

    for name in names:
        I.append(cv.imread(dir+'\\'+name, -1))
        
    I = np.asarray(I)
    
    mean, std = square_roi_statistic(I, wavelenght, 506)

    return mean, std

def generate_source_graph():
    current_dir = os.getcwd() 
    data_path = current_dir + '\\dataset\\Hemoglobin\\Caracterization\\2022.09.22\\position\\'

    file_name = 'ls_caracterization_'

    position = [-2, -1, 0, 1, 2, 3];

    mean = []
    stdev = []

    for i in range(1, 7):
        dir = data_path + '0' + str(i)
        mean_vec, std_vec = region_mean(dir, data_path)
        mean.append(mean_vec)
        stdev.append(std_vec)
    
    dir = data_path + '01'
    wavelength = extract_wavelenght_from_name(dir,data_path)

    norm_mean = np.divide(mean, 255)
    norm_stdev = np.divide(stdev, 255)
    
    t_mean = np.transpose(norm_mean)
    t_stdev = np.transpose(norm_stdev)


    plot_save_figure(position,t_mean[wavelength.index(506)], t_stdev[wavelength.index(506)], 4,
                     np.linspace(-4, 4, 500), "Posição [cm]", [-4, 4, 0, 0.5], file_name + 'pos.png')

    plot_save_figure(wavelength, norm_mean[position.index(0)], norm_stdev[position.index(0)], 9,
                     np.linspace(400, 600, 2000), "Comprimento de onda [nm]", [400, 600, 1e-3, 2], file_name + 'wl.png')

def plot_save_figure(x_data, y_data, y_data_err, degree, x_fit, label_x, axis, file_name):
    current_dir = os.getcwd() 
    dst_folder = current_dir + '\\results\\'
    scr_folder = current_dir + '\\'

    coeff = np.polyfit(x_data,y_data, degree)
    y = np.poly1d(coeff)
    
    plt.rcParams['font.size'] = 24
    plt.figure(figsize=(12,9));
    plt.errorbar(x_data, y_data, yerr = y_data_err, fmt='o', color= 'black', 
                ecolor = 'red', elinewidth = 1, capsize = 2, label='Dados coletados')
    if label_x == "Comprimento de onda [nm]":
        plt.semilogy(x_fit,y(x_fit),'--k', label='Curva de tendêcia')
    else:
        plt.plot(x_fit,y(x_fit),'--k', label='Curva de tendêcia')
    plt.ylabel("Reflectância difusa normalizada")
    plt.xlabel(label_x)
    plt.axis(axis)
    plt.legend()
    plt.savefig(file_name)
    plt.show()
    if os.path.exists(dst_folder + file_name):
        path = dst_folder + file_name
        os.remove(path)
        shutil.move(scr_folder + file_name, dst_folder + file_name)
    else:
        shutil.move(scr_folder + file_name, dst_folder + file_name)


def extract_wavelenght_from_name(dir:str, path):
    
    names = os.listdir(dir)
    position = dir.replace(path,'')
    wl = []
    for data in names:
        temp = data.replace('.bmp', '')
        temp = temp.replace('cat_'+ position + '_','')
        wl.append(int(temp))
    return wl

generate_source_graph()

