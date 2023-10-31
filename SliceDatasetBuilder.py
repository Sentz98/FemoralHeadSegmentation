import os
import glob
import torch
import json
import pydicom
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# The number of images to exclude from the positive interval
exclusionInterval = 20

class CustomHipDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, view_types=None, transform=None):
        """
        Args:
            json_path (string): Path to the JSON file with annotations.
            view_types (list, optional): List of view types to include in the dataset. Defaults to ["axial", "coronal", "sagittal"].
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform

        with open(json_path, 'r') as file:
            data_info = json.load(file)

        self.image_paths = []
        self.labels = []

        if view_types == None:
            view_types = ["axial", "coronal", "sagittal"]
        for view_type_key in view_types:
            if view_type_key not in ["axial", "coronal", "sagittal"]:
                raise ValueError("Invalid view type. Use 'axial', 'coronal', or 'sagittal'.")

        for subj_info in data_info:
            subj_name = subj_info["subj"]
            for view_type in view_types:
                image_dir = os.path.join(subj_info["folder"], subj_name, view_type)

                print(f"Loading {image_dir}...")

                # Determine if the view type differentiates between DX and SX
                sides = ["SX", "DX"] if view_type in ["axial", "coronal"] else [None]

                for side_idx, side in enumerate(sides):
                    positive_interval = subj_info[view_type][side_idx]
                    print(f"Positive interval of {subj_name}: {positive_interval} for {side}")
                    excluded_interval = [
                        max(0, positive_interval[0] - exclusionInterval),
                        positive_interval[1] + exclusionInterval
                    ]

                    for image_name in os.listdir(image_dir):
                        # Validate if the image matches the side for non-sagittal types
                        if side and not image_name.endswith(f"_{side}.png"):
                            continue

                        # Extract the image number
                        img_num = int(image_name.split('_')[1].split('.')[0])

                        # Exclude images within the excluded interval
                        if excluded_interval[0] <= img_num <= positive_interval[0] or positive_interval[1] <= img_num <= excluded_interval[1]:
                            continue

                        # Label as positive if within the positive interval
                        label = positive_interval[0] <= img_num <= positive_interval[1]

                        self.image_paths.append(os.path.join(image_dir, image_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB").convert("L")

        if self.transform is not None:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        return image, label
    
    def load_all_dataset(self):
        self.__images__ = []
        self.__labels__ = []
        for i in tqdm(range(self.__len__()), "Load all dataset"):
            img, label = self.__getitem__(i)
        
            self.__images__.append(img)
            self.__labels__.append(label)
        
        return
    
    def get_all_dataset(self):
        return self.__images__, self.__labels__

# Usage:
# dataset = CustomHipDataset("your_json_path.json")
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

#--------------------------------------------------------------------------------------------------------------

# Function to create png images from dicom files
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
    load_and_save("data/normalHip/JOR09")
    



