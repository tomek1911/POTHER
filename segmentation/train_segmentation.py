from comet_ml import Experiment
import os
import numpy as np
import argparse
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold

from segmentation.models.unet import UNet
from segmentation.data_loader import LungSegDataset
from segmentation.loss_functions import DiceLoss

parser = argparse.ArgumentParser(description="Training Segmentation Network on Fetal Dataset.")
parser.add_argument("--images",
                    type=str,
                    default="data/segmentation/images",
                    help="Path to the images")
parser.add_argument("--masks",
                    type=str,
                    default="data/segmentation/masks",
                    help="Path to the masks")
parser.add_argument("--validation_split",
                    type=float,
                    default=0.2,
                    help="Validation split")
parser.add_argument("--in_channels",
                    type=int,
                    default=1,
                    help="Number of input channels")
parser.add_argument("--epochs",
                    type=int,
                    default=100,
                    help="Number of epochs")
parser.add_argument("--img_size",
                    type=int,
                    default=512,
                    help="X image size")
parser.add_argument("--num_workers",
                    type=int,
                    default=0,
                    help="Number of workers for processing the data")
parser.add_argument("--classes",
                    type=int,
                    default=1,
                    help="Number of classes in the dataset")
parser.add_argument("--batch_size",
                    type=int,
                    default=2,
                    help="Number of batch size")
parser.add_argument("--lr",
                    type=float,
                    default=0.0001,
                    help="Number of learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.0001,
                    help="Number of weight decay")
parser.add_argument("--display_steps",
                    type=int,
                    default=20,
                    help="Display steps in console")
parser.add_argument("--backbone",
                    type=str,
                    default="resnet152",
                    help="Encoder backbone")
parser.add_argument("--parallel",
                    type=bool,
                    default=False,
                    help="Parallel learning on GPU")
parser.add_argument("--GPU",
                    type=bool,
                    default=True,
                    help="Use GPU")
parser.add_argument("--model_name",
                    type=str,
                    default="cxr_model",
                    help="Model name")
args = parser.parse_args()

experiment = Experiment("uicx0MlnuGNfKsvBqUHZjPFQx")
experiment.log_parameters(args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kfold = KFold(n_splits=5, shuffle=False)

criterion = DiceLoss()

train_dataset = LungSegDataset(path_to_images=args.images,
                               path_to_masks=args.masks,
                               image_size=args.img_size,
                               mode="train")

validation_dataset = LungSegDataset(path_to_images=args.images,
                                    path_to_masks=args.masks,
                                    image_size=args.img_size,
                                    mode="valid")

print("---------------")


def jaccard(outputs, targets):
    """

    Args:
        outputs:
        targets:

    Returns:

    """
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (outputs * targets).sum(1)
    union = (outputs + targets).sum(1) - intersection
    jac = (intersection + 0.001) / (union + 0.001)
    return jac.mean()


loss_min = np.inf

# Start time of learning
total_start_training = time.time()


for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f"FOLD: {fold}")
    print("-------------")

    train_sampler = SubsetRandomSampler(train_ids)
    val_sampler = SubsetRandomSampler(val_ids)

    train_loader = DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               sampler=train_sampler)

    val_loader = DataLoader(validation_dataset,
                            batch_size=args.batch_size,
                            sampler=val_sampler)

    model = UNet(input_channels=1,
                 output_channels=64,
                 n_classes=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0001,
                                 weight_decay=0.0001
                                 )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, min_lr=1e-9)

    with experiment.train():
        best_val_score = 0.0

        for epoch in range(args.epochs):
            start_time_epoch = time.time()
            print('Starting epoch {}/{}'.format(epoch + 1, args.epochs))
            # train
            model.train()
            running_loss = 0.0
            running_jaccard = 0.0
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs_masks = model(images)
                loss_seg = criterion(outputs_masks, masks)
                loss = loss_seg
                loss.backward()
                optimizer.step()

                jac = jaccard(outputs_masks.round(), masks)
                running_jaccard += jac.item()
                running_loss += loss.item()

                if batch_idx % args.display_steps == 0:
                    mask = masks[0, 0, :]
                    out = outputs_masks[0, 0, :]
                    res = torch.cat((mask, out), 1).cpu().detach()
                    experiment.log_image(res, name=f"Train: {batch_idx}/{epoch}")
                    print('    ', end='')
                    print('batch {:>3}/{:>3} loss: {:.4f}, Jaccard {:.4f}, learning time:  {:.2f}s\r' \
                          .format(batch_idx + 1, len(train_loader),
                                  loss.item(), jac.item(),
                                  time.time() - start_time_epoch))

            # evalute
            print('Finished epoch {}, starting evaluation'.format(epoch + 1))
            model.eval()
            val_running_loss = 0.0
            val_running_jac = 0.0
            for batch_idx, (image, mask) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs_masks = model(images)
                loss_seg = criterion(outputs_masks, masks)
                loss = loss_seg

                val_running_loss += loss.item()
                jac = jaccard(outputs_masks.round(), masks)
                val_running_jac += jac.item()

                if batch_idx % args.display_steps == 0:
                    mask = masks[0, 0, :]
                    out = outputs_masks[0, 0, :]
                    res = torch.cat((mask, out), 1).cpu().detach()
                    experiment.log_image(res, name=f"Val: {batch_idx}/{epoch}")

            train_loss = running_loss / len(train_loader)
            val_loss = val_running_loss / len(val_loader)

            train_jac = running_jaccard / len(train_loader)
            val_jac = val_running_jac / len(val_loader)

            save_path = f"trained_models/model-fold-{fold}_{args.img_size}"

            if best_val_score < val_jac:
                torch.save(model.state_dict(), save_path)
                best_val_score = val_jac
                print(f"Current best val score {best_val_score}. Model saved!")

            scheduler.step(val_loss)

            experiment.log_current_epoch(epoch)
            experiment.log_metric("train_jac", train_jac)
            experiment.log_metric("val_jac", val_jac)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metric("val_loss", val_loss)
            experiment.log_metric("train_jac", train_jac)
            experiment.log_metric("val_jac", val_jac)

            print('    ', end='')
            print('loss: {:.4f}  jaccard: {:.4f} \
                        val_loss: {:.4f} val_jac: {:4.4f}\n' \
                  .format(train_loss, train_jac, val_loss, val_jac))

    print('Training UNet finished, took {:.2f}s'.format(time.time() - total_start_training))

