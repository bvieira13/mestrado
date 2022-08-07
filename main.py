import os
import numpy as np
import imfun as im
import matplotlib.pyplot as plt;
import shutil

def equalize_histogram(img):
  # Computando o valo máximo e mínimo dos niveis de cinza presentes na imagem
  max = np.max(img);
  min = np.min(img);
  # Computando os valores da nova imagem após a transformação
  img_norm = (img.astype(np.float32) - min)*255/(max-min);
  img_out = img_norm.astype(np.uint8);
  return img_out

def load_data(file, n):
    
    return data
def espectral_square_roi(hsi_img, index = 25):
    
    [img_mask, points] = im.polyroi(hsi_img[index])

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

def espectral_roi(hsi_img, index = 23):
    
    [img_mask, points] = im.polyroi(hsi_img[index]);
    img_mask = img_mask[:,:];
    img_mask[img_mask>0] = 1;

    vec_espctral = [];

    for m in range(0, len(hsi_img)):
        img_temp = img_mask*hsi_img[m][:,:];
        mean = np.mean(img_temp[img_temp>0]);
        vec_espctral.append(mean);
    
    return vec_espctral;

def main():
    current_dir = os.getcwd();
    wavelength = np.linspace(400,720,33);

    img_dir = current_dir + '\dataset\Hand';
    ref_dir = current_dir + '\dataset\Reference';

    hsi_img = im.load_gray_images(img_dir);
    hsi_ref = im.load_gray_images(ref_dir);    

<<<<<<< HEAD
    ref_espectral = espectral_roi(hsi_ref);
=======
    ref_espectral = espectral_square_roi(hsi_ref);
>>>>>>> e73fcd4fab71b10cfbc3d2b6095530bfcacecbbb
    img_espectral = espectral_roi(hsi_img);


    white_espectral = np.divide(ref_espectral, ref_espectral); 
    hand_espectral = np.divide(img_espectral, ref_espectral);
    hand_norm = [];

    for k in range(0, len(hsi_img)):
        norm = hsi_img[k][:,:]/ref_espectral[k];
        hand_norm.append(norm);

    file_name = 'hsi_reflectance_norm.png'
    dst_folder = current_dir + '\\results\\';
    scr_folder = current_dir + '\\';

    plt.figure(figsize=(5,5));
<<<<<<< HEAD
    plt.semilogy(wavelength, white_espectral, '-b', label="Reflectância de branco puro");
    plt.semilogy(wavelength, hand_espectral, '-r', label="Reflectância de pele");
    plt.title("Curva espectral de reflectância");
=======
    plt.semilogy(wavelength, white_espectral, '-b', label='Branco');
    plt.semilogy(wavelength, hand_espectral, '-r', label='Tecido cutâneo');
    plt.title('Curva espectral de tecido cutâneo');
    plt.xlabel('Comprimento de onda (nm)');
    plt.ylabel('Reflectância normalizada')
>>>>>>> e73fcd4fab71b10cfbc3d2b6095530bfcacecbbb
    plt.legend();
    plt.savefig(file_name);
    plt.show();

    if os.path.exists(dst_folder + file_name):
        path = dst_folder + file_name;
        os.remove(path)
        shutil.move(scr_folder + file_name, dst_folder + file_name)
    else:
        shutil.move(scr_folder + file_name, dst_folder + file_name)

main()