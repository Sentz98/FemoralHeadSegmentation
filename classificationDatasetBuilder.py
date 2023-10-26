import os
import glob
import pydicom
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_and_save(filepath, roi= None):
    # create the destination folder
    if not os.path.exists("data/dataset"):
        os.mkdir("data/dataset")
        os.mkdir("data/dataset/normalHip")
        os.mkdir("data/dataset/dysplasticHip")
        os.mkdir("data/dataset/retrovertedHip")

    if filepath.split('/')[-2] == "normalHip":
        dest_folder = "data/dataset/normalHip"
    elif filepath.split('/')[-2] == "dysplasticHip":
        dest_folder = "data/dataset/dysplasticHip"
    elif filepath.split('/')[-2] == "retrovertedHip":
        dest_folder = "data/dataset/retrovertedHip"
    else:
        print("Error: folder not found")
        return    
    
    subject = filepath.split('/')[-1]
    dest_folder = f"{dest_folder}/{subject}"
        
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
        os.mkdir(f"{dest_folder}/axial")
        os.mkdir(f"{dest_folder}/coronal")
        os.mkdir(f"{dest_folder}/sagittal")

    # load the image files
    image_files = glob.glob(os.path.join(filepath, '*.dcm'))
    image_files.sort()

    Im = []

    for k in tqdm(range(1, len(image_files)), desc=f"Processing {subject} DICOMs"):
        ds = pydicom.dcmread(image_files[k - 1])
        im = ds.pixel_array  # Load pixel data
        Im.append(im)
 
    # Convert the list of NumPy arrays to a NumPy array
    Im = np.array(Im)

    # If roi is not specified, start routine to view the image and select the roi
    if roi is None:
        roi = [[0, Im.shape[0]], [0, Im.shape[1]], [0, Im.shape[2]]]

    # save the axial view image
    for i in tqdm(range(roi[0][0], roi[0][1]), desc="Saving axial slices"):
        axial = Im[i, :, :]
        #split image in two parts (left and right)
        axialSX = axial[:, :axial.shape[1]//2]
        axialDX = axial[:, axial.shape[1]//2:]

        plt.imsave(f"{dest_folder}/axial/{subject}_{i}_SX.png", axialSX, cmap='gray')
        plt.imsave(f"{dest_folder}/axial/{subject}_{i}_DX.png", axialDX, cmap='gray')

    # save the coronal view image
    for i in tqdm(range(roi[1][0], roi[1][1]), desc="Saving coronal slices"):
        coronal = np.rot90(np.rot90(Im[:, i, :]))
        #split image in two parts (left and right)
        coronalSX = coronal[:, :coronal.shape[1]//2]
        coronalDX = coronal[:, coronal.shape[1]//2:]

        plt.imsave(f"{dest_folder}/coronal/{subject}_{i}_SX.png", coronalSX, cmap='gray')
        plt.imsave(f"{dest_folder}/coronal/{subject}_{i}_DX.png", coronalDX, cmap='gray')
    
    # save the sagittal view image
    for i in tqdm(range(roi[2][0], roi[2][1]), desc="Saving sagittal slices"):
        sagittal = np.rot90(np.rot90(Im[:, :, i]))
        plt.imsave(f"{dest_folder}/sagittal/{subject}_{i}.png", sagittal, cmap='gray')
    
    print(f"{subject} dataset saved in {dest_folder}")

    


#def function to load image and save slices in dataset folder

if __name__ == "__main__":
    load_and_save("data/normalHip/JOR01")
    load_and_save("data/normalHip/JOR02")
    load_and_save("data/dysplasticHip/TRAD09_3x_full")
    load_and_save("data/dysplasticHip/TRAD10_3x_full")
    load_and_save("data/retrovertedHip/OAC01")
    load_and_save("data/retrovertedHip/OAC02")
    



