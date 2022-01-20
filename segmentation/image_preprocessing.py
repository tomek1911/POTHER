import numpy as np
import os
from glob import glob

image_path = os.path.join("data/segmentation/images/")
mask_path = os.path.join("data/segmentation/masks/")

images = os.listdir(image_path)
mask = os.listdir(mask_path)

images.sort()
mask.sort()

mask = [fName.split(".png")[0] for fName in mask]
image_file_name = [fName.split("_mask")[0] for fName in mask]

check = [i for i in mask if "mask" in i]
testing_files = set(os.listdir(image_path)) & set(os.listdir(mask_path))
training_files = check

