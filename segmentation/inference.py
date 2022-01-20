"""Inference for Chest X-Ray dataset."""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from segmentation.models.unet import UNet
import glob


class ChestXRInference:
    """EchoCardioInference class."""

    def __init__(self, model_path: str = None
                 ) -> None:
        """
        Args:
            model_path:
        """
        self.model_path = model_path
        self.model = UNet(input_channels=1,
                          output_channels=64,
                          n_classes=1)
        self.model.load_state_dict(torch.load(self.model_path,
                                              map_location="cpu"))
        self.model.eval()

        self.transforms = transforms.Compose([
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
        pred_mask = transforms.ToPILImage()(data).resize(size)
        ig = plt.imshow(pred_mask)
        plt.show()

    def save_predictions(self,
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
        pred_mask = transforms.ToPILImage()(data).resize(size)
        split_path = image_name.split(os.sep)
        to_folder = split_path[5]
        study_instanceUID = split_path[6]
        id = split_path[7]
        filename = split_path[8][:-4]
        if not os.path.exists(f"/Volumes/Seagate/Kaggle_COVID/predictions768/{to_folder}/{study_instanceUID}/{id}"):
            os.makedirs(f"/Volumes/Seagate/Kaggle_COVID/predictions768/{to_folder}/{study_instanceUID}/{id}")
        pred_mask.save(f"/Volumes/Seagate/Kaggle_COVID/predictions768/{to_folder}/{study_instanceUID}/{id}/{filename}.png")


if __name__ == "__main__":
    lung_seg = ChestXRInference(model_path="trained_models/lung_seg.pt")
    test_path = "/Volumes/Seagate/Kaggle_COVID/train_png/train/*/*/*.png"

    for i, file in enumerate(glob.glob(test_path)):
        folder = os.path.basename(file).split("_")[0]
        filename = os.path.basename(file)
        #lung_seg.get_visual_prediction(image_name=file)
        lung_seg.save_predictions(image_name=file)
