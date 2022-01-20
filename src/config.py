import os
import cv2
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform, to_tuple
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src import utils
import multiprocessing
import numpy as np
import random

gpus_id = {'0': 'Nvidia Titan RTX - 24GB', '1': 'Nvidia Titan Xp - 12GB'}
comp_type = A.transforms.ImageCompression.ImageCompressionType.JPEG

# CUSTOM AUGMENTATIONS
def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    out = img[offset:]
    return out


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    out = img[offset_h:offset_h + size, offset_w:offset_w + size]
    return out


def central_percentage_crop(img, percent):
    height, width = img.shape
    # height, width, _ = img.shape # RGB ver
    x_vector = int(width/2*percent)
    y_vector = int(height/2*percent)
    # out = img[x_vector:(width-2*x_vector), y_vector:(width-2*y_vector), :] # RGB
    out = img[x_vector:(width-2*x_vector), y_vector:(width-2*y_vector)]
    return out


class TopCrop(DualTransform):

    def __init__(self, percent, always_apply=False, p=1):
        super(TopCrop, self).__init__(always_apply, p)
        self.percent = percent

    def apply(self, img, **params):
        return crop_top(img, percent=self.percent)


class CentralCrop(DualTransform):
    def __init__(self, always_apply=False, p=1):
        super(CentralCrop, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return central_crop(img)


class CentralPercentageCrop(DualTransform):
    def __init__(self, percent, always_apply=False, p=1):
        super(CentralPercentageCrop, self).__init__(always_apply, p)
        self.percent = percent

    def apply(self, img, **params):
        return central_percentage_crop(img, self.percent)

class RandomCLAHE(ImageOnlyTransform):
    
    def __init__(self, clip_range=[2, 4], grid_range=[8.0,12.0], always_apply=False, p=0.5):
        super(RandomCLAHE, self).__init__(always_apply, p)
        self.clip_range = clip_range
        self.grid_range = grid_range

    def apply(self, img, clip_limit=2.0, tile_grid_size = (8,8), **params):
        return clahe(img, clip_limit, tile_grid_size)

    def get_params(self):
        # return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}
        grid_size = int(random.uniform(self.grid_range[0], self.grid_range[1]))
        grid_size = (grid_size,grid_size)
        return {"clip_limit": random.uniform(self.clip_range[0], self.clip_range[1]), 
                "tile_grid_size": grid_size}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


# CONFIG
class Config:
    def __init__(self, args):

        models_dir = args.dir

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        self.size = args.input_size
        self.transform_type = args.augmentation

        #CONFIG HARDWARE
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        cpu_count = multiprocessing.cpu_count()
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() and args.device != -1 else torch.device("cpu")
        if self.device.type == 'cuda':
            self.gpu_info = gpus_id[args.device]
            print(
                f'\n{__class__.__name__} - using device: {self.device}, {self.gpu_info}, {cpu_count} cores cpu.')
        elif self.device.type == 'cpu':
            self.gpu_info = f"{cpu_count} cores cpu"
            print(
                f'\n{__class__.__name__} - Using device: {self.device}, {cpu_count} cores cpu.')

        ##AUGMENTATIONS
        if self.transform_type == "torchvision":
            self.transform_train = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                utils.normalize])
            self.transform_test = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                utils.normalize])
        elif self.transform_type == "albumentations":

            self.preproc_transform = A.Compose([
                TopCrop(0.05, always_apply=True),
                CentralCrop(always_apply=True),
                CentralPercentageCrop(percent = 0.15, always_apply=True),
            ])

            self.geom_transform_train = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=5, interpolation=cv2.INTER_LINEAR,
                                   border_mode=0, value=0, p=0.5),
                A.augmentations.geometric.transforms.Affine(scale={'x':(0.9,1.1), 'y':(0.9,1.1)}, p=0.5),
                A.augmentations.transforms.HorizontalFlip(),
            ])

            self.value_transform_train = A.Compose([
                RandomCLAHE(clip_range=[2.0, 4.0],
                            grid_range=[4.0, 8.0], p=0.25),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.25),
                A.Sharpen(p=0.25),
                A.Emboss(p=0.25)
            ]
            )
            self.transform_train = A.Compose([
                A.Resize(self.size, self.size, cv2.INTER_LINEAR),
                self.value_transform_train,
                self.geom_transform_train
            ])
            
            self.transform_mask_train = A.Compose([
                A.Resize(self.size, self.size, cv2.INTER_NEAREST),
                self.geom_transform_train
            ])

            self.transform_test = A.Compose([
                A.Resize(self.size, self.size)
            ])
        elif self.transform_type == "strong_augment":
            self.transform_train = A.Compose([
                ### RESIZE AND CROP 
                # Global training
                # TopCrop(0.08, always_apply=True),
                # CentralCrop(always_apply=True),
                # CentralPercentageCrop(percent = 0.15, always_apply=True),                        
                A.Resize(self.size, self.size, always_apply=True),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, interpolation=1,
                                      border_mode=0, value=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.05, p=0.25),
                A.OneOf([
                    A.augmentations.geometric.transforms.ElasticTransform(),
                    A.augmentations.transforms.GridDistortion(num_steps = 5, distort_limit = 0.3)
                ]),
                A.augmentations.transforms.Equalize(by_channels=False, p=1), 
                A.augmentations.transforms.HorizontalFlip(p=0.5),
                A.augmentations.transforms.VerticalFlip(p=0.5),
                A.OneOf([
                    A.augmentations.transforms.GaussianBlur(blur_limit=(3,11)),
                    A.MotionBlur(),
                    A.MedianBlur(blur_limit=7),
                    A.Blur(blur_limit=9)
                ]),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.IAAPiecewiseAffine(p=0.3)
                ]),
                A.OneOf([
                    A.Sharpen(),
                    A.Emboss(),
                ]),
                A.OneOf([
                    A.CLAHE(clip_limit=4,p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=0.5),
                ]),
                A.OneOf([
                    A.GaussNoise(),
                    A.augmentations.transforms.ImageCompression(quality_lower = 75, quality_upper= 100)
                ]),
            ])
            self.transform_test = A.Compose([
                ### RESIZE AND CROP 
                # Global training
                # TopCrop(0.08, always_apply=True),
                # CentralCrop(always_apply=True),
                # CentralPercentageCrop(percent = 0.15, always_apply=True), 
                A.Resize(self.size, self.size)
            ])
            