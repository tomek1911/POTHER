from scipy.sparse import data
from src.params import args, parser
import argparse

import json 
with open("models/last_parser_params.json", "w") as outfile:
        json.dump(vars(args), outfile, indent=4)

import os
os.environ["COMET INI"] = os.path.join(os.getcwd(),".comet.config")
os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

exp = None
if args.comet:
    from comet_ml import Experiment, ConfusionMatrix
    exp = Experiment()
    tags = args.tags.split('_')
    tags += [args.model, args.dataset]
    exp.add_tags(tags)


import numpy as np
import random
import torch
import torchvision.models as models 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import WeightedRandomSampler
from datetime import datetime

from src import train
from src import DiceLoss, WeightedSoftDiceLoss, FocalLoss, DiceCoefficient, L1ReconstructionLoss, L2ReconstructionLoss
from src import V7DarwinDataset, CovidXDataset, WangBenchmarkDataset, BIMCVDataset
from src import Config

from multitask_unet.unet import Unet

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# reproducibility
if args.seed != -1:
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.det:
        torch.use_deterministic_algorithms(True)
else:
    g=None

conf = Config(args)
date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

if args.patch_preview:   
    args.visualise = True
    args.bs=1

if args.dataset == 'darwin':
    dataset_train = V7DarwinDataset(args, conf, mode='train')
    dataset_val = V7DarwinDataset(args, conf, mode='val')
    # dataset_test = V7DarwinDataset(args, conf, mode='test')
elif args.dataset == 'covidx':
    dataset_train = CovidXDataset(args, conf, mode='train')
    if args.validation:
        dataset_val = CovidXDataset(args, conf, mode='val')
    else:
        dataset_val = CovidXDataset(args, conf, mode='test')
elif args.dataset == 'wang':
    dataset_train = WangBenchmarkDataset(args, conf, mode='train')
    dataset_val = WangBenchmarkDataset(args, conf, mode='val')
elif args.dataset == 'bimcv':
    dataset_train = BIMCVDataset(args, conf, mode='train')
    dataset_val = BIMCVDataset(args, conf, mode='val')

print(f"datasets: train: {len(dataset_train)}, val: {len(dataset_val)}")

if args.weights != 'none':
    weights = dataset_train.weights

    #weighted sampler
    weights = [weight / max(weights) for weight in weights]
    print(weights)
    weights_dataset = [weights[label] for label in dataset_train.labels]
    sampler = WeightedRandomSampler(weights_dataset, len(dataset_train), generator=g, replacement = True)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, sampler=sampler,
                                            num_workers=args.workers, worker_init_fn=seed_worker,
                                            generator=g)
else:
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs,
                                               num_workers=args.workers, shuffle=True, worker_init_fn=seed_worker,
                                               generator=g)


val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.bs,
                                         num_workers=args.workers, shuffle=False)     


#DATASET PREVIEW
if args.patch_preview:                           
    import cv2
    from src.utils import tensor_img2np
    from src import draw_utils
    matrix = []
    horizontal_patches = []
    for batch_idx, (img, mask, label, pb, idx) in enumerate(train_loader):
        if batch_idx < 30:
            org = cv2.imread(dataset_train.img_paths[idx[0]], cv2.IMREAD_GRAYSCALE)
            pb = torch.cat(pb).detach().cpu().tolist()
            tl = (pb[3], pb[0])
            br = (pb[1], pb[2])
            draw_utils.drawrect(org, tl, br, color = 255, thickness=2, style='dashed')
            
            img = tensor_img2np(img.detach().cpu(), denorm=True)
            mask = tensor_img2np(mask.detach().cpu())
            lung = (img*(mask/255)).astype(np.uint8)

            if len(horizontal_patches) !=5:
                horizontal_patches.append(img)
            else:
                matrix.append(cv2.hconcat(horizontal_patches))
                horizontal_patches = []
            folder = "patches_equ"
            if len(matrix) == 5:
                image = cv2.vconcat(matrix)
                cv2.imwrite(f"{folder}/all_patches.png", image)   
                break

            cv2.imwrite(f"{folder}/{batch_idx:2d}_patch_context.png", org)    
            cv2.imwrite(f"{folder}/{batch_idx:2d}_image_patch.png", img)        
            cv2.imwrite(f"{folder}/{batch_idx:2d}_mask_patch.png", mask)  
            cv2.imwrite(f"{folder}/{batch_idx:2d}_lung.png", lung)   
        else:
            break                               

#TRAINING PARAMS
print("\n*ALL HYPERPARAMETERS*\n")
args_dict = vars(args)
args_dict['device_name'] = conf.gpu_info
{print(f'> {k:<25}-- {str(v):<25}') for (k,v) in args_dict.items()}

#MODEL
tasks={'T1':args.T1, 'T2':args.T2, 'T3':args.T3}
print("\n*UNET SETUP*<\n")
model = Unet(backbone_name=args.model, pretrained=args.ptr, segmentation_classes=1, decoder_filters=(1024, 512, 256, 128, 64),
             classes_encoder=args.classes_num, infer_tensor=torch.zeros(1,1,args.input_size,args.input_size),
             decoder_use_batchnorm = True, attention=args.use_attention,
             experimental_head=args.experimental_head, norm_features=args.norm_features, tasks=tasks)

model.to(conf.device)

#load pretrained model for unet backbone
if  args.use_ptr_bb:
    print("Loading pretrained backbone for unet")
    pretrained_bb = torch.load(args.pretrained_bb_path)
    loaded_pretrained_model = pretrained_bb['model']
    if args.classes_num != 5:
        loaded_pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    pretrained_dict = loaded_pretrained_model.state_dict()
    #filter keys
    model_dict = model.backbone.state_dict()
    if args.classes_num == 5:
        new_dict = {k.strip('backbone').lstrip('.'): v for k, v in pretrained_dict.items() if k.strip('backbone').lstrip('.') in model_dict} # take all features from backbone
    # new_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc.weight', 'fc.bias'] and k in model_dict}
    else:
        new_dict = {k: v for k, v in pretrained_dict.items() if 'layer4' in k and k in model_dict}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    model_dict.update(new_dict)
    model.backbone.load_state_dict(model_dict)

if exp:
    exp.set_model_graph(model, overwrite=True)

#load model to continue training
if args.resume != "":
    trained_model = torch.load(args.resume)
    #load arguments and update current args with previous
    t_args = argparse.Namespace()
    args_dict = trained_model['parameters']
    t_args.__dict__.update(args_dict)
    epochs = args.epochs
    for key, value in vars(t_args).items():
        if key in vars(args):
            setattr(args, key, value)
        else:
            vars(args)[key]=value
    args.epochs = epochs
    args.prev_epochs=int(args.resume.split('/')[-1].rstrip('.pth').split('_')[-1])
    {print(f'> {k:<25}-- {str(v):<25}') for (k,v) in vars(args).items()}
    model = trained_model['model']
    model.to(conf.device)

if args.freeze_features:
    for name, param in model.named_parameters(): 
        if name in ['classification_head.fc.weight', 'classification_head.fc.bias']:
            param.requires_grad = True 
        else:
            param.requires_grad = False

# if args.freeze_layers:
#     #freeze initial layers from pretrained model
#     for name, param in model.named_parameters(): 
#             if any(layer in name for layer in ['layer1', 'layer2', 'layer3', 'layer4']):
#                 param.requires_grad = False 
#             else:
#                 param.requires_grad = True

# LOSS FUNCTION
# >classification<
if args.weights != 'none':
    weights = torch.as_tensor(weights, dtype=torch.float).to(conf.device)
    loss_class_fn = nn.CrossEntropyLoss(weight = weights, reduction='mean').to(conf.device)
else:
    if args.classes_num !=2:
        loss_class_fn = nn.CrossEntropyLoss(reduction='mean').to(conf.device)
    else:
        loss_class_fn = nn.BCEWithLogitsLoss(reduction='mean').to(conf.device)

# >segmentation<
# loss_mask_fn = WeightedSoftDiceLoss(conf.device, v1=0.05, v2=0.95).to(conf.device)
loss_seg_fn = DiceLoss().to(conf.device)
loss_rec_fn = L1ReconstructionLoss().to(conf.device)
# loss_mask_fn2 = FocalLoss(conf.device, alpha=0.8, gamma=2.0).to(conf.device)
dice = DiceCoefficient().to(conf.device)

#OPTIMIZER
optimizer = optim.RAdam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
# optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
# optimizer = optim.SGD(params=model.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.wd)

scheduler = None
if args.sched:
    scheduler = MultiStepLR(optimizer, milestones=args.sched_steps, gamma = args.sched_gamma, verbose = True)
    print(f"Using multistep scheduler with milestones at: {args.sched_steps}.")

train(args, model, optimizer, scheduler, dataset_train, dataset_val, train_loader, val_loader,
      conf, tasks, loss_class_fn, loss_seg_fn, loss_rec_fn, dice, exp)