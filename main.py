import os
import numpy as np
import imfun as im
import matplotlib.pyplot as plt;

def equalize_histogram(img):
  # Computando o valo máximo e mínimo dos niveis de cinza presentes na imagem
  max = np.max(img);
  min = np.min(img);
  # Computando os valores da nova imagem após a transformação
  img_norm = (img.astype(np.float32) - min)*255/(max-min);
  img_out = img_norm.astype(np.uint8);
  return img_out

def espectral_roi(hsi_img, index = 25):
    
    ref_img_roi, points = im.polyroi(hsi_img[index])

    dim_x = [];
    dim_y = [];

    for i in range(0, len(points)):
        (x,y) = points[i];
        dim_x.append(x);
        dim_y.append(y);
    
    xmin = np.min(dim_x);
    xmax = np.max(dim_x);

    ymin = np.min(dim_y);
    ymax = np.max(dim_y);

    vec_espctral = [];

    for m in range(0, len(hsi_img)):
        crop_roi = hsi_img[m][xmin:xmax,ymin:ymax];
        mean = np.mean(crop_roi);
        vec_espctral.append(mean);
    
    return vec_espctral;

def main():
    current_dir = os.getcwd();
    wavelength = np.linspace(400,720,33);

    img_dir = current_dir + '\Hand';
    ref_dir = current_dir + '\Reference';

    hsi_img = im.load_gray_images(img_dir);
    hsi_ref = im.load_gray_images(ref_dir);    

    ref_espectral = espectral_roi(hsi_ref);
    img_espectral = espectral_roi(hsi_img);


    white_espectral = np.divide(ref_espectral, ref_espectral); 
    hand_espectral = np.divide(img_espectral, ref_espectral);

    plt.figure(figsize=(5,5));
    plt.semilogy(wavelength, white_espectral, '-b', label="Reflectância de branco puro");
    plt.semilogy(wavelength, hand_espectral, '-r', label="Reflectância de pele");
    plt.title("Curva espectral de reflectância");
    plt.legend();
    plt.show();


main()