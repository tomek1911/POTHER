import enum
from unittest.mock import patch
from src.params import args
from itertools import chain
from torchvision.models import resnet
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as f
import itertools

from datetime import datetime
from collections import Counter

from PIL import Image
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam

from src import draw_utils
from src import V7DarwinDataset, CovidXDataset, BIMCVDataset
from src import Config
from src.utils import freeze_layers, tensor2PIL_1ch, tensor2PIL, inv_norm_1CH

conf = Config(args)
date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
if int(args.device) != -1:
    trained_model = torch.load(args.trained_model)
else:
    trained_model = torch.load(args.trained_model, map_location=torch.device('cpu'))
model = trained_model['model']
model.to(conf.device)

#dataset = V7DarwinDataset(args, conf, mode='test')
dataset = CovidXDataset(args, conf, mode='test')

loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, num_workers=args.workers, pin_memory=True)

dataset_len = len(dataset)
loader_len = len(loader)
dataset_classes_count = dataset.classes_count
date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

predictions_all = [[]for i in range(dataset_len)]
predictions = []

model.eval()
torch.set_grad_enabled(True)

## GRADCAM TARGETS AND FOLDERS - POTHER
main_grad_folder = "votes_area" 
# main_grad_folder = "gradcam_patch_mesh"
# grad_folder = "ag_conv_layer1"
# target=model.upsample_blocks[2].conv.conv[0]
# grad_folder = "ag_conv_layer2"
# target=model.upsample_blocks[1].conv.conv[0]
# grad_folder = "ag_conv_layer3"
# target = model.upsample_blocks[0].conv.conv[0]
# grad_folder = "ag_layer1"
# target=model.upsample_blocks[2].att.att_out
# grad_folder = "ag_layer2"
# target = model.upsample_blocks[1].att.att_out
grad_folder = "ag_layer3"
target = model.upsample_blocks[0].att.att_out
# grad_folder = "attention_skip_layer3_pneumonia"

## GLOBAL TRAINING GRADCAM TARGET
# target = model.backbone.layer4[-1].conv3
# grad_folder = "layer4_conv3"

gradcam = GradCAM.from_config(model_type='resnet', arch=model, target_layer=target) #layer_name="layer4_bottleneck2_conv3"

def jetmap(img : np.array) -> np.array:
    img = (img * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img

def jetmap_normalize(img : np.array) -> np.array:
    img /= img.max()
    img = (img * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img

def mark_class_patch(image, patch_pos, votes):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for id, (pb, vote) in enumerate(zip(patch_pos,votes)):
        tl = (pb[3], pb[0])
        br = (pb[1], pb[2])
        color = (255,255,255)
        if vote == "normal":
            color = (0,255,0)
        elif vote == "pneumonia":
            color = (0,0,255)
        elif vote == "COVID-19":
            color = (255,0,0)
        draw_utils.drawrect(image, tl, br, color = color, thickness=2, style='dashed', gap = 10)
        # cv2.putText(image, str(id), (int(pb[1]-0.15*args.patch_base*2),int(pb[2]-0.05*args.patch_base*2)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def draw_patch_context(image_id, patch_positions_list, dataset):
    original_img = Image.open(dataset.img_paths[image_id]).convert('L')
    original_img = np.array(original_img, dtype=np.uint8)
    original_mask = Image.open(dataset.mask_paths[image_id]).convert('RGB')
    original_mask = np.array(original_mask, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for id, pb in enumerate(patch_positions_list): # top, left, bottom, right
        tl = (pb[3], pb[0])
        br = (pb[1], pb[2])
        center = (int((pb[3] - pb[1])/2) + pb[1],int((pb[2] - pb[0])/2)+pb[0])
        draw_utils.drawrect(original_img, tl, br, color = 255, thickness=3, style='dashed', gap = 8)
        draw_utils.drawrect(original_mask, tl, br, color = (255,0,0), thickness=3, style='dashed', gap = 8)
        cv2.circle(original_mask,center,2,(255,0,0),-1)
        # cv2.putText(original_img, str(id), (int(pb[1]-0.2*args.patch_base*2),int(pb[2]-0.05*args.patch_base*2)), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(original_mask, str(id), (int(pb[1]-0.2*args.patch_base*2),int(pb[2]-0.05*args.patch_base*2)), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    
    return original_img, original_mask

def is_patch_center_in_draw_area(pc, draw_area):
    if draw_area[pc[1]][pc[0]] == 255:
        return True
    return False


def get_patch_centers(center, p_base, x0=-1, y0=-1, xn=1, yn=1):
    cx = center[0]
    cy = center[1]
    xx, yy = np.meshgrid(np.arange(x0, xn+1), np.arange(y0, yn+1), sparse=False)
    xx_p = p_base * xx + cx
    yy_p = p_base * yy + cy
    xxs = list(itertools.chain(*xx_p.tolist()))
    yys = list(itertools.chain(*yy_p.tolist()))
    centers = [(x,y) for x,y in zip(xxs,yys)]
    return centers

    ## aggregate area of voting based on mask non-zero pixels
    
    # coverage_container = np.zeros((1024, 1024))
    # patch_region = ((pb[2]-pb[0]), (pb[1]-pb[3]))
    # mask = cv2.resize(np.squeeze(mask.numpy()), patch_region[::-1], interpolation=cv2.INTER_LINEAR)
    # coverage_container[pb[0]:pb[2], pb[3]:pb[1]] = np.multiply(np.ones(patch_region, dtype=int), mask)  
    # container[idx]+=coverage_container  

votes = args.votes_count
classes = args.classes_num 
folder = 'covidx_activations'

#DARWIN indices
no_pneumonia = [8, 13, 15, 24]
pneumonia = [0,1,2,19]
covid = [3,10,35,97,159]

#COVIDX indices
no_pneumonia = [10, 20, 30, 40]
pneumonia = [140,150,160,170]
covid = [199,210,220,230]

vis_patches = False
agreagate = False

s=5
patch_centers = get_patch_centers((500,500), 80, x0=-s,y0=-s, xn=s, yn=s)
# for figures [210, 222, 279, 212, 2, 10, 123, 160, 163, 102]
for dataset_image_id in [216]:
    print(f"img: id - {dataset_image_id}")
    canvas = [np.zeros((1024, 1024)) for i in range(classes)]
    container = [np.zeros((1024, 1024)) for i in range(classes)]
    original_images = []
    activation_max = []
    
    draw_area = np.array(dataset.__getitem__(dataset_image_id)[5])
    # draw_area = cv2.dilate(draw_area, np.ones((21, 21), 'uint8'))
    patch_centers_filtered = [pc for pc in patch_centers if is_patch_center_in_draw_area(pc, draw_area)]
    for i in range(1):# for idx in range(classes):

        gradcam_target = ''
        ground_truth = ''
        patches_positions = []
        activations = []
        patch_votes = []

        for idx in range(votes):
        # for idx, p_center in enumerate(patch_centers_filtered):

            img, mask, label, pb, _, _ = dataset.__getitem__(dataset_image_id)
            # img, mask, label, pb, _, _ = dataset.__getitem__(dataset_image_id, p_center)
            # img, mask, label, pb = external_dataset.__getitem__(dataset_image_id, p_center)
            img = img.to(conf.device, dtype=torch.float32).unsqueeze(0)
            mask = mask.to(conf.device, dtype=torch.float32).unsqueeze(0)
            patches_positions.append(pb)

            if not vis_patches:
                x_class, x_seg, x_res = model(img)
                pred = x_class.argmax().item()
                voted_class = dataset.class_inv_dict[pred]
                patch_votes.append(voted_class) 
                # segmentation = torch.sigmoid(x_seg)

                target=label
                gradcam_target = dataset.class_inv_dict[target]
                out_path = os.path.join(main_grad_folder,grad_folder,f'{gradcam_target}_img_{dataset_image_id}')
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                activation, _ = gradcam(img, class_idx=target)
                img = inv_norm_1CH(img)
                heatmap, overlay = visualize_cam(activation, img, alpha=0.7)
                overlay_np8 = (overlay.permute(1,2,0).numpy()*255).astype(np.uint8)
                if args.patch_base != 112:
                    ts = args.patch_base * 2
                    out_img = cv2.resize(overlay_np8, (ts,ts),interpolation=cv2.INTER_LINEAR)
                    activations.append(out_img)
                else:
                    activations.append(overlay_np8)
                img = img.squeeze().detach().cpu().numpy()*255
                mask = mask.squeeze().detach().cpu().numpy()*255
                Image.fromarray(img).convert('L').save(f'{out_path}/grad_{idx}_src.png')
                Image.fromarray(mask).convert('L').save(f'{out_path}/grad_{idx}_src_mask.png')
                Image.fromarray(overlay_np8).convert('RGB').save(f'{out_path}/grad_{idx}.png')

            # probability weighted gradcams
            if agreagate:
                #squeeze, detach, make 3channels, to numpy, add to container
                orginal_img = Image.open(dataset.img_paths[dataset_image_id]).convert('RGB')
                original_images.append(torch.tensor(np.array(orginal_img)).permute(2, 0, 1).float().div(255))

                #get probability of target class
                prob = f.softmax(x_class, dim = 1)
                prob = prob[0][target].item()

                # to numpy
                activation_np = np.squeeze(activation.detach().cpu().numpy())
                # segmentation_np = np.squeeze(segmentation.detach().cpu().numpy())

                # 2d numpy array - (row, col) = (y,x)
                patch_region = ((pb[2]-pb[0]), (pb[1]-pb[3]))

                # opencv 2d array  (cols, rows) = (x,y)
                activation_np = cv2.resize(activation_np, patch_region[::-1], interpolation=cv2.INTER_LINEAR) * prob
                # segmentation_np = cv2.resize(segmentation_np, patch_region[::-1], interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(np.squeeze(mask.numpy()), patch_region[::-1], interpolation=cv2.INTER_LINEAR)

                #accumulate activations
                activation_container = np.zeros((1024, 1024))
                # activation_container[pb[0]:pb[2], pb[3]:pb[1]] = np.multiply(activation_np, segmentation_np)
                activation_container[pb[0]:pb[2], pb[3]:pb[1]] = np.multiply(activation_np, mask)
                canvas[idx] += activation_container  

                #accumulate coverage
                coverage_container = np.zeros((1024, 1024))
                coverage_container[pb[0]:pb[2], pb[3]:pb[1]] = np.multiply(np.ones(patch_region, dtype=int), mask)  
                container[idx]+=coverage_container  
        
                gradcam_target = dataset.class_inv_dict[target]
                ground_truth = dataset.class_inv_dict[label]
                
                print(f"\rGenerate activation maps for class - {target} vs {label} - {idx+1}/{classes}: {i+1:3d}/{votes}", end="")
        
        img_pb, mask_pb = draw_patch_context(dataset_image_id, patches_positions, dataset)
        
        Image.fromarray(img_pb).save(os.path.join(out_path,'source_image.png'))
        Image.fromarray(mask_pb).save(os.path.join(out_path,'source_mask.png'))
        
        activations_img = Image.open(dataset.img_paths[dataset_image_id]).convert('RGB')
        activations_img = np.array(activations_img).astype(np.uint8)
        for (pb, ovr) in zip(patches_positions, activations):
            height = pb[2] - pb[0]
            width = pb[1] - pb[3]
            if height != width:
                ovr = cv2.resize(ovr,(width, height))
            activations_img[pb[0]:pb[2], pb[3]:pb[1], :] = ovr
        activations_img = mark_class_patch(activations_img, patches_positions, patch_votes)

        Image.fromarray(activations_img).save(os.path.join(out_path,'activations_image.png'))
        [print(f"{idx}-{vote}") for idx, vote in enumerate(patch_votes)]
        
        #activation_max.append(canvas[idx].max())

        #Voting analysis, area, coverage 

        # freq_container = container[0].astype(np.uint8)
        # mask = cv2.imread(dataset.mask_paths[15], cv2.IMREAD_GRAYSCALE)
        # mask_indices = np.nonzero(mask) # 0 - row_indices(y), 1 - col_indices(x)
        # lung_votes_vector = freq_container[mask_indices]
        # freq_mean = lung_votes_vector.mean()
        # votes_sum = lung_votes_vector.sum()
        # std_lung = np.std(lung_votes_vector)

        # mask_area = len(freq_container[mask_indices])
        # lung_votes_vector = freq_container[mask_indices]
        # non_zero_pixels = np.nonzero(lung_votes_vector)[0]
        # std_non_zero = np.std(lung_votes_vector[non_zero_pixels])
        # freq_non_zero_pixels = len(non_zero_pixels)
        # freq_zero_pixels_count = mask_area - freq_non_zero_pixels

        # print("VOTING summary:")
        # print(f"\t>Votes: {votes}")
        # print(f"\t>Max overlapping votes: {freq_container.max()}")
        # print(f"\t>Std all votes: {std_lung:.3f}")
        # print(f"\t>Std non-zero votes: {std_non_zero:.3f}")
        # print(f"\t>Mean votes: {freq_mean:.3f}")
        # print(f"\t>Votes sum: {votes_sum}")
        # print(f"\t>Lung mask area: {mask_area}")
        # print(f"\t>Image area: {mask.size}")
        # print(f"\t>Pixel count without vote: {freq_zero_pixels_count}")
        # print(f"\t>Unvoted/All ratio: {freq_zero_pixels_count/mask_area:.3f}")

        # max_container = cv2.equalizeHist(freq_container)
        # heatmap = cv2.applyColorMap(max_container, cv2.COLORMAP_JET)
        # cv2.imwrite(f'votes_analysis/coverage_{votes}_non_zero.png', max_container)
        # cv2.imwrite(f'votes_analysis/coverage_{votes}_non_zero_heat.png', heatmap)
        # cv2.imwrite(f'votes_analysis/cov_pos_{votes}.png', orginal_img)
        # cv2.imwrite(f'votes_analysis/cov_pos_mask_{votes}.png', orginal_mask)
        # stop = 0

    # print(activation_max)
    # scale_factors = [item / max(activation_max) for item in activation_max]
    # print(scale_factors)

    ## probability weighted GRADCAMS, normalised with max activation

    # for i in range(3):
    #     gradcam_target = dataset.class_inv_dict[i]
    #     # activation weighted by number votes per pixel
    #     norm_canv1 = np.divide(canvas[i], container[i] + 1e-9)
    #     heatmap, overlay = visualize_cam(torch.tensor(norm_canv1), original_images[i], alpha=1.0)
    #     # activation weighted by max activation of all targets for single image
    #     norm_canv2 = norm_canv1 / norm_canv1.max()
    #     norm_canv2 *= scale_factors[i]

    #     #to uint8 for save
    #     overlay_np = (overlay.permute(1,2,0).numpy()*255).astype(np.uint8)
    #     canv = jetmap_normalize(norm_canv1)
    #     canv_norm = jetmap(norm_canv2)

    #     #save to folder
    #     Image.fromarray(canv).convert('RGB').save(f'{folder}/{dataset_image_id:03d}_activation_{ground_truth}_{gradcam_target}.png')
    #     Image.fromarray(canv_norm).convert('RGB').save(f'{folder}/{dataset_image_id:03d}_activation_{ground_truth}_{gradcam_target}_norm.png')
    #     Image.fromarray(overlay_np).save(f'{folder}/{dataset_image_id:03d}_activation_{ground_truth}_{gradcam_target}_overlay.png')
