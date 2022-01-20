import os
from unittest.mock import patch
import torch
import csv
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from src.config import Config
import random
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

ImageFile.LOAD_TRUNCATED_IMAGES = True

transform_image = A.Compose([
A.Normalize(mean=[0.5], std=[0.5]),
ToTensorV2()
])
transform_mask = ToTensorV2()

def scale_cnt(cnt, scale=0.75):
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            cnt_norm = cnt - [cX, cY]
            cnt_scaled = cnt_norm * scale
            cnt_scaled = cnt_scaled + [cX, cY]
            cnt_scaled = cnt_scaled.astype(np.int32)
              
            return cnt_scaled

def get_ENS_weights(num_classes, samples_per_class, beta=0.999999):
    ens = 1.0 - np.power([beta]*num_classes,
                         np.array(samples_per_class, dtype=np.float))
    weights = (1.0-beta) / np.array(ens)
    weights = weights / np.sum(weights) * num_classes
    ens = torch.as_tensor(weights, dtype=torch.float)
    return ens

def load_bounding_boxes(path):
    with open(path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = list(reader)
    return [[float(val) for val in record[1:-2]] for record in data[1:]]

# PATCH LEARNING
class PatchSampler():
    def __init__(self, args, mode) -> None:

        self.patch_base = args.patch_base
        self.visualise = args.visualise
        self.random_patch_size = args.random_patch_size
        self.use_reduced_draw = args.use_reduced_draw
        self.simple_area_draw = args.simple_area_draw
        self.non_zero_mask_draw = args.non_zero_mask_draw
        self.patch_base_mult = args.patch_base_mult
        self.non_random_patch = args.non_random_patch
        self.crop_ratio = args.crop_ratio
        self.mode = mode

    # for visualisation
    def take_patch(self, image, mask, center, draw_area=None):

        px = center[0]
        py = center[1]
        p_base = self.patch_base
        image_max = image.shape[0]
    
        top= max(0, py-p_base)
        bottom = min(py+p_base, image_max)
        left = max(0, px-p_base)
        right = min(px+p_base, image_max)

        patch_img = image[top:bottom,
                          left:right]
        patch_mask = mask[top:bottom,
                          left:right]

        return patch_img, patch_mask, [top, right, bottom, left]

    # for training
    def get_patch(self, image, mask, draw_area=None):

        image_max = image.shape[0]

        #PATCH BASE SIZE
        if self.random_patch_size:
            p_base = int(random.uniform(self.patch_base, self.patch_base_mult * self.patch_base))
        else:
            p_base = self.patch_base

        #PATCH CENTER POSITION - X & Y
        patch_centers = None
        if self.non_zero_mask_draw:
            patch_centers = np.nonzero(mask)
        elif self.use_reduced_draw:
            patch_centers = np.nonzero(draw_area)
        elif self.simple_area_draw:
            #assumption that mask and image are square 1:1
            center = int(image.shape[0]/2.0)
            vector = int(center * (1.0 - self.crop_ratio))
            min_pos = vector
            max_pos = 2*center - vector
            px = int(random.uniform(min_pos, max_pos))
            py = int(random.uniform(min_pos, max_pos))

        if patch_centers:
            if len(patch_centers[0]) != 0:
                indices_length = len(patch_centers[0])
                index = random.randint(0,indices_length-1)
                px = patch_centers[1][index]
                py = patch_centers[0][index]
       
        patch_img = image[max(0, py-p_base):min(py+p_base, image_max),
                          max(0, px-p_base):min(px+p_base, image_max)]
        patch_mask = mask[max(0, py-p_base):min(py+p_base, image_max),
                          max(0, px-p_base):min(px+p_base, image_max)]

        if self.visualise:
            top= max(0, py-p_base)
            bottom = min(py+p_base, image_max)
            left = max(0, px-p_base)
            right = min(px+p_base, image_max)
            return patch_img, patch_mask, [top, right, bottom, left] #, (px,py)

        return patch_img, patch_mask, []


class V7DarwinDataset(Dataset):

    def __init__(self, args, config, mode='train'):

        # 3 classes version
        if args.classes_num == 3:
            self.classes_names = ["no_finding", "pneumonia", "covid-19"]
            self.class_dict = {"no_finding": 0, "pneumonia": 1, "covid-19": 2}
            self.class_inv_dict = {0: "no_finding", 1: "pneumonia", 2: "covid-19"}
            self.classes_count = 3

        #4 classes version
        if args.classes_num == 4:
            self.classes_names = ["no_finding", "bacterial_pneumonia", "viral_pneumonia", "covid-19"]
            self.class_dict = {"no_finding": 0, "bacterial_pneumonia": 1, "viral_pneumonia": 2, "covid-19": 3}
            self.class_inv_dict = {0: "no_finding", 1: "bacterial_pneumonia", 2: "viral_pneumonia", 3: "covid-19"}
            self.classes_count = 4


        self.transform_type = 'albumentations'
        self.mode = mode
        self.data_path = args.data_path
        self.input_size = args.input_size
        self.patch = args.use_patch
        self.visualise = args.visualise
        if self.patch:
            self.patch_sampler = PatchSampler(args, mode)

        #SETUP DATASET CSV FILE PATHS
        if mode == 'train':
            self.transform = config.transform_train
            self.dataset_csv = args.dataset_csv + "_train.csv"
        elif mode == "val":
            self.transform = config.transform_test
            self.dataset_csv = args.dataset_csv + "_val.csv"
        elif mode == 'test':
            self.transform = config.transform_test
            self.dataset_csv = args.dataset_csv + "_test.csv"

        #IMAGES, MASKS and LABELS csv based
        self.dataset_info_df = pd.read_csv(self.dataset_csv)
        self.img_paths = [os.path.join(self.data_path, 'images', filename)
                          for filename in self.dataset_info_df['file_name'].tolist()]
        self.mask_paths = [os.path.join(self.data_path, 'masks', filename)
                           for filename in self.dataset_info_df['mask'].tolist()]
        self.draw_area_paths = [os.path.join(self.data_path, 'masks_draw', filename)
                           for filename in self.dataset_info_df['mask'].tolist()]
        self.labels = [self.class_dict[label]
                       for label in self.dataset_info_df['class'].tolist()]
        # self.bounding_boxes = load_bounding_boxes(args.bb_file)


        #CALCULATE WEIGHTS 
        if mode == 'train':
            self.samples_per_class = [
                self.labels.count(i) for i in set(self.labels)]
            if args.weights == 'ens':
                self.weights = get_ENS_weights(self.classes_count, self.samples_per_class, beta=args.ens_beta)
            elif args.weights == 'inverse_freq': 
                self.weights = [1.0/weight for weight in self.samples_per_class]
            else:
                self.weights = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        label = self.labels[idx]
        image = Image.open(self.img_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        draw_area = Image.open(self.draw_area_paths[idx]).convert('L')

        if self.patch:
            image, mask, boundries = self.patch_sampler.get_patch(np.array(image), np.array(mask), np.array(draw_area))
        else:
            image = np.array(image)
            mask = np.array(mask)

        if self.transform_type == 'albumentations':
            transformed_dict = self.transform(image=image, mask=mask)
            image = transform_image(image=transformed_dict['image'])['image']
            mask = transform_mask(image=transformed_dict['mask'])['image'].float()/255.0

        if self.visualise and self.patch:
            return image, mask, label, boundries

        return image, mask, label

class CovidXDataset(Dataset):

    def __init__(self, args, config, mode='train'):

        # 3 classes version
        if args.classes_num == 3:
            self.classes_names = ["normal", "pneumonia", "COVID-19"]
            self.class_dict = {"normal": 0, "pneumonia": 1, "COVID-19": 2}
            self.class_inv_dict = {0: "normal", 1: "pneumonia", 2: "COVID-19"}
            self.classes_count = 3
        if args.classes_num == 5:
            self.classes_names = ["rsna", "cohen", "sirm", "fig1","actmed"]
            self.class_dict = {"rsna": 0, "cohen": 1, "sirm": 2, "fig1":3, "actmed":4}
            self.class_inv_dict = {0:"rsna", 1:"cohen", 2:"sirm", 3:"fig1", 4:"actmed"}
            self.classes_count = 5

        self.transform_type = 'albumentations'
        self.mode = mode
        self.data_path= args.data_path_covidx
        self.input_size = args.input_size
        self.patch = args.use_patch
        self.visualise = args.visualise
        if self.patch:
            self.patch_sampler = PatchSampler(args, mode)

        #SETUP DATASET CSV FILE PATHS
        if args.validation and mode == 'train':
            self.transform = config.transform_train
            self.dataset_csv = args.dataset_csv_covidx + "_train2.csv"
            #self.dataset_csv = args.dataset_source_task_path + "_train.csv" # dataset source test
        elif not args.validation and mode == 'train':
            self.transform = config.transform_train
            self.dataset_csv = args.dataset_csv_covidx + "_train.csv"
        if mode == "val":
            self.transform = config.transform_test
            self.dataset_csv = args.dataset_csv_covidx + "_val.csv"
            #self.dataset_csv = args.dataset_source_task_path + "_val.csv"
            mode = 'train' # images are in train folder
        if mode == 'test':
            self.transform = config.transform_test
            # self.dataset_csv = args.dataset_csv_covidx + "_test.csv"
            self.dataset_csv = '../CovidXBenchmark/benchmark_data/test_covidx.csv'

        #IMAGES, MASKS and LABELS csv based
        self.dataset_info_df = pd.read_csv(self.dataset_csv)
        img_folder = '_1024_equ' if args.equ_dataset else '_1024' 
        if self.classes_count == 5:
            lbl_column = 'data_source'
        else:
            lbl_column = 'class'

        if args.nolungs:
            img_folder = '_no_lungs'
        if args.onlylungs:
            img_folder = '_only_lungs'
        if args.masklungs:
            img_folder = '_masks_filtered_interpolated'

        self.img_paths = [os.path.join(self.data_path, f'{mode}{img_folder}', filename)
                          for filename in self.dataset_info_df['filename'].tolist()]
        self.mask_paths = [os.path.join(self.data_path, f'{mode}_masks_filtered_interpolated', filename)
                           for filename in self.dataset_info_df['filename'].tolist()]
        self.draw_area_paths = [os.path.join(self.data_path, f'{mode}_masks_reduced9', filename)
                            for filename in self.dataset_info_df['filename'].tolist()]
  
        self.labels = [self.class_dict[label]
                       for label in self.dataset_info_df[lbl_column].tolist()]
        # self.bounding_boxes = load_bounding_boxes(args.bb_file)

        if mode == 'train':
            self.samples_per_class = [
                self.labels.count(i) for i in set(self.labels)]
            if args.weights == 'ens':
                self.weights = get_ENS_weights(self.classes_count, self.samples_per_class, beta=args.ens_beta)
            elif args.weights == 'inverse_freq': 
                self.weights = [1.0/weight for weight in self.samples_per_class]
            else:
                self.weights = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx, patch_center = None):
            
        label = self.labels[idx]
        image = Image.open(self.img_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        draw_area = Image.open(self.draw_area_paths[idx]).convert('L')

        if self.patch:
            if patch_center is not None:
                image, mask, boundries = self.patch_sampler.take_patch(np.array(image), np.array(mask), patch_center, np.array(draw_area))
            else:
                image, mask, boundries = self.patch_sampler.get_patch(np.array(image), np.array(mask), np.array(draw_area))
        else:
            image = np.array(image)
            mask = np.array(mask)

        if self.transform_type == 'albumentations':
            transformed_dict = self.transform(image=image, mask=mask)
            image = transform_image(image=transformed_dict['image'])['image']
            mask = transform_mask(image=transformed_dict['mask'])['image'].float()/255.0

        if self.visualise and self.patch:
            return image, mask, label, boundries, idx, draw_area

        return image, mask, label

class WangBenchmarkDataset(Dataset):

    def __init__(self, args, config, mode='train'):

        # 3 classes version
        if args.classes_num == 3:
            self.classes_names = ["normal", "pneumonia", "COVID-19"]
            self.class_dict = {"normal": 0, "pneumonia": 1, "COVID-19": 2}
            self.class_inv_dict = {0: "normal", 1: "pneumonia", 2: "COVID-19"}
            self.classes_count = 3

        self.transform_type = 'albumentations'
        self.mode = mode
        self.data_path= args.data_path_wang
        self.input_size = args.input_size
        self.patch = args.use_patch
        self.visualise = args.visualise
        if self.patch:
            self.patch_sampler = PatchSampler(args, mode)

        #SETUP DATASET CSV FILE PATHS
        if mode == 'train':
            self.transform = config.transform_train
            self.dataset_csv = args.dataset_csv_wang + "_train.csv"
        elif mode == "val":
             self.transform = config.transform_test
             self.dataset_csv = args.dataset_csv_wang + "_val.csv"
        elif mode == 'test':
            self.transform = config.transform_test
            self.dataset_csv = args.dataset_csv_wang + "_test.csv"

        #IMAGES, MASKS and LABELS csv based
        self.dataset_info_df = pd.read_csv(self.dataset_csv)
        self.img_paths = [os.path.join(self.data_path, 'images_equalized', filename)
                          for filename in self.dataset_info_df['filename'].tolist()]
        self.mask_paths = [os.path.join(self.data_path, 'masks_interpolated', filename)
                           for filename in self.dataset_info_df['filename'].tolist()]
        self.draw_area_paths = [os.path.join(self.data_path, 'masks_draw', filename)
                            for filename in self.dataset_info_df['filename'].tolist()]
        self.labels = [self.class_dict[label]
                       for label in self.dataset_info_df['class'].tolist()]
        # self.bounding_boxes = load_bounding_boxes(args.bb_file)

        if mode == 'train':
            self.samples_per_class = [
                self.labels.count(i) for i in set(self.labels)]
            if args.weights == 'ens':
                self.weights = get_ENS_weights(self.classes_count, self.samples_per_class, beta=args.ens_beta)
            elif args.weights == 'inverse_freq': 
                self.weights = [1.0/weight for weight in self.samples_per_class]
            else:
                self.weights = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
            
        label = self.labels[idx]
        image = Image.open(self.img_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        draw_area = Image.open(self.draw_area_paths[idx]).convert('L')

        if self.patch:
            image, mask, boundries = self.patch_sampler.get_patch(np.array(image), np.array(mask), np.array(draw_area))
        else:
            image = np.array(image)
            mask = np.array(mask)

        if self.transform_type == 'albumentations':
            transformed_dict = self.transform(image=image, mask=mask)
            image = transform_image(image=transformed_dict['image'])['image']
            mask = transform_mask(image=transformed_dict['mask'])['image'].float()/255.0

        if self.visualise and self.patch:
            return image, mask, label, boundries

        return image, mask, label

class BIMCVDataset(Dataset):

    def __init__(self, args, config, mode='train'):

        # 2 classes version
        if args.classes_num == 2:
            self.classes_names = ["cov_negative", "cov_positive"]
            self.class_dict = {"cov_negative": 0, "cov_positive": 1}
            self.class_inv_dict = {0: "cov_negative", 1: "cov_positive"}
            self.classes_count = 2

        self.transform_type = 'albumentations'
        self.mode = mode
        self.data_path= args.data_path_bimcv
        self.input_size = args.input_size
        self.patch = args.use_patch
        self.visualise = args.visualise
        if self.patch:
            self.patch_sampler = PatchSampler(args, mode)

        #SETUP DATASET CSV FILE PATHS
        if mode == 'train':
            self.transform = config.transform_train
            self.dataset_csv = args.dataset_csv_bimcv + "_train.csv"
        elif mode == "val":
             self.transform = config.transform_test
             self.dataset_csv = args.dataset_csv_bimcv + "_val.csv"
        elif mode == 'test':
            self.transform = config.transform_test
            self.dataset_csv = args.dataset_csv_bimcv + "_test.csv"

        #IMAGES, MASKS and LABELS csv based
        self.dataset_info_df = pd.read_csv(self.dataset_csv)
        self.img_paths = [os.path.join(self.data_path, 'bimcv_images', filename)
                           for filename in self.dataset_info_df['filename'].tolist()]
        self.mask_paths = [os.path.join(self.data_path, 'masks_interpolated', filename)
                           for filename in self.dataset_info_df['filename'].tolist()]
        # self.draw_area_paths = [os.path.join(self.data_path, 'masks_draw', filename)
        #                     for filename in self.dataset_info_df['filename'].tolist()]
        self.draw_area_paths = None
        self.labels = self.dataset_info_df['class'].tolist()
        # self.bounding_boxes = load_bounding_boxes(args.bb_file)

        if mode == 'train':
            self.samples_per_class = [
                self.labels.count(i) for i in set(self.labels)]
            if args.weights == 'ens':
                self.weights = get_ENS_weights(self.classes_count, self.samples_per_class, beta=args.ens_beta)
            elif args.weights == 'inverse_freq': 
                self.weights = [1.0/weight for weight in self.samples_per_class]
            else:
                self.weights = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx, patch_center=None):
            
        label = self.labels[idx]
        image = Image.open(self.img_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.draw_area_paths:
            draw_area = Image.open(self.draw_area_paths[idx]).convert('L')

        if self.patch:
            if patch_center is not None:
                image, mask, boundries = self.patch_sampler.take_patch(np.array(image), np.array(mask), patch_center)
            else:
                image, mask, boundries = self.patch_sampler.get_patch(np.array(image), np.array(mask))
        else:
            image = np.array(image)
            mask = np.array(mask)

        if self.transform_type == 'albumentations':
            transformed_dict = self.transform(image=image, mask=mask)
            image = transform_image(image=transformed_dict['image'])['image']
            mask = transform_mask(image=transformed_dict['mask'])['image'].float()/255.0

        if self.visualise and self.patch:
            return image, mask, label, boundries

        return image, mask, label

    
