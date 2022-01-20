"""Inference for Chest X-Ray dataset."""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.unet import UNet
import glob
import argparse
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='CXR image segmentation inference script')

parser.add_argument('--model_path', type = str, default = 'segmentation/models/lung_seg.pt', help = 'This is the dir for the output masks')
parser.add_argument('--input_dir', type = str, default = 'BIMCV/covid_positive_filtered_1024_8bit', help = 'This is the path of the input data')
parser.add_argument('--output_dir', type = str, default = 'BIMCV/covid_positive_masks', help = 'This is the dir for the output masks')
parser.add_argument('--size', type = int, default = 1024, help = 'Size of the output masks')
parser.add_argument('--batch', type = int, default = -1, help = 'Amount of images in the folder to segment - for -1 will run for all')
parser.add_argument('--first_img', type = int, default = 0, help = 'Id of image to start with.')
parser.add_argument('--device', type = str, default = 'cuda', choices=['cuda', 'cpu'], help = 'Amount of images in the folder to segment - for -1 will run for all')

args = parser.parse_args()

class ChestXRInference:
    """EchoCardioInference class."""

    def __init__(self, model_path: str = None, device = 'cpu'
                 ) -> None:
        """
        Args:
            model_path:
        """
        self.model_path = model_path
        self.model = UNet(input_channels=1,
                          output_channels=64,
                          n_classes=1)

        self.device=device

        if self.device == 'cuda':
            self.model.cuda()

        self.model.load_state_dict(torch.load(self.model_path,
                                              map_location=device))                                              
        self.model.eval()        

        self.transforms = transforms.Compose([
            transforms.Resize((224,224), interpolation=0),
            transforms.ToTensor()
        ])

    def get_visual_prediction(self,
                              image_name
                              ):
        """
        Args:
            image_name:
        Returns:
        """
        image = Image.open(image_name).convert("L")

        size = (768, 768)
        img = self.transforms(image).unsqueeze(0)
        pred_mask = self.model(Variable(img))
        data = pred_mask.round().squeeze(0).cpu().data
        pred_mask = transforms.ToPILImage()(data).resize(size, Image.NEAREST)
        ig = plt.imshow(pred_mask)
        plt.show()

    def save_predictions(self,
                         image_name,
                         out_size,
                         out_dir
                         ):
        """
        Args:
            image_name:
        Returns:
        """
        image = Image.open(image_name).convert("L")

        size = (out_size, out_size)
        img = self.transforms(image).unsqueeze(0)
        img = img.to(device=self.device)
        pred_mask = self.model(Variable(img))
        data = pred_mask.round().squeeze(0).cpu().data
        pred_mask = transforms.ToPILImage()(data).resize(size)
        split_path = image_name.split(os.sep)       
        filename = split_path[-1]       
        pred_mask.save(os.path.join(out_dir,filename))



lung_seg = ChestXRInference(model_path=args.model_path, device=args.device)
images= sorted(glob.glob(os.path.join(args.input_dir,"*.*")))

if args.batch == -1: 
    images = images[args.first_img:]
else:
    images = images[args.first_img:args.batch]

print("")
print(f" > Starting segmentation of: {len(images)} images, on device: {args.device}. Initial image id: {args.first_img}.")
print(f" > input dir: {args.input_dir},")
print(f" > output dir: {args.output_dir},")
print(f" > output size of masks: {args.size}.")

time_start0 = time.time()

for i, file in enumerate(images):
    time_start = time.time()
    lung_seg.save_predictions(image_name=file, out_size=args.size, out_dir=args.output_dir)

    print(f'\rSegmented and saved {i+1}/{len(images)} img(s), time per img: {time.time()-time_start:.2f}s.', end="")

print("")
print(f"Finished. It took {time.time() - time_start0:.0f}s.")