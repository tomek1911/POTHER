from src.params import args, parser
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from itertools import chain

from datetime import datetime
from collections import Counter

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from src import V7DarwinDataset, CovidXDataset, WangBenchmarkDataset
from src import Config
from src.utils import analyse_confusion_matrix

conf = Config(args)
date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# LOAD SAVED MODEL STATE
trained_model = torch.load(args.trained_model)

#LOAD HYPERPARAMETERS AND FLAGS
t_args = argparse.Namespace()
args_dict = trained_model['parameters']
t_args.__dict__.update(args_dict)
epochs = args.epochs
votes = args.votes_count
model_path = args.trained_model
dataset = args.dataset
for key, value in vars(t_args).items():
    if key in vars(args):
        setattr(args, key, value)
    else:
        vars(args)[key]=value
args.dataset = dataset
args.epochs = epochs
args.votes_count = votes
args.bs=12
args.workers=12
args.trained_model = model_path
{print(f'> {k:<25}-- {str(v):<25}') for (k,v) in vars(args).items()}

#LOAD MODEL
model = trained_model['model']
model.to(conf.device)

#CHOOSE DATASET

if args.dataset == 'darwin':
    dataset = V7DarwinDataset(args, conf, mode='test')
elif args.dataset == 'covidx':
    dataset = CovidXDataset(args, conf, mode='test')
elif args.dataset == 'wang':
    dataset = WangBenchmarkDataset(args, conf, mode='test')

loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, num_workers=args.workers)

model.eval()
torch.set_grad_enabled(False)

dataset_len = len(dataset)
loader_len = len(loader)
dataset_classes_count = dataset.classes_count
date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

predictions_all = [[]for i in range(dataset_len)]

#######################################################################
# # MESH VOTING
# def is_patch_center_in_draw_area(pc, draw_area):
#     if draw_area[pc[1]][pc[0]] == 255:
#         return True
#     return False

# def get_patch_centers(center, p_base, x0=-1, y0=-1, xn=1, yn=1):
#     cx = center[0]
#     cy = center[1]
#     xx, yy = np.meshgrid(np.arange(x0, xn+1), np.arange(y0, yn+1), sparse=False)
#     xx_p = p_base * xx + cx
#     yy_p = p_base * yy + cy
#     xxs = list(chain(*xx_p.tolist()))
#     yys = list(chain(*yy_p.tolist()))
#     centers = [(x,y) for x,y in zip(xxs,yys)]
#     return centers

# s=6
# patch_centers = get_patch_centers((512,512), args.patch_base*2, x0=-s,y0=-s, xn=s, yn=s)
# final_pred = []
# gt = []

# for dataset_image_id in range(len(dataset)):

#     #filter patches outside fraw area
#     draw_area = np.array(dataset.__getitem__(dataset_image_id)[5])
#     filtered_patch_centers = [pc for pc in patch_centers if is_patch_center_in_draw_area(pc, draw_area)]
#     predictions = []

#     for idx, p_center in enumerate(filtered_patch_centers):
    
#         img, _, label, _ , _, _ = dataset.__getitem__(dataset_image_id, p_center)
#         img = img.to(conf.device, dtype=torch.float32).unsqueeze(0)
#         target = label
#         x_class, _, _ = model(img)
        
#         predictions.append(x_class.argmax().item())
        
#         print(f"\rinference:  voting: {idx+1}/{len(filtered_patch_centers)}, {dataset_image_id+1}/{len(dataset)}", end="")
    
#     gt.append(target)
#     votes = Counter(predictions).most_common(n=1)
#     pred = votes[0][0]
#     final_pred.append(pred)

# print(f"\nFINAL VOTING BASED ON THE MESH OF PATCHES")
# print(f"Used model:{args.trained_model}")
# print(classification_report(gt,final_pred,target_names=dataset.classes_names, digits=3))
# print(confusion_matrix(gt, final_pred))
#######################################################################

# RANDOM VOTING
global_gt = None

for i in range(args.votes_count):
    
    #probabilities for metrics
    probs = np.zeros((dataset_len, dataset_classes_count), dtype = np.float32)             
    gt    = np.zeros(dataset_len, dtype = np.float32)
    k=0
    predictions = []

    for batch_idx, (img, mask, label) in enumerate(loader):

        img = img.to(conf.device, dtype=torch.float32)
        mask = mask.to(conf.device, dtype=torch.float32)
        target = label.to(conf.device, dtype=torch.long)

        x_class, x_seg, x_res = model(img)
        # segm_activation = torch.sigmoid(x_seg)

        predictions.append(x_class.max(1).indices.tolist())

        probs[k: k + x_class.shape[0], :] = f.softmax(x_class, dim = 1).detach().cpu()
        gt[   k: k + x_class.shape[0]] = target.detach().cpu()
        k += x_class.shape[0]
        print(f"\rinference:  voting: {i+1}/{args.votes_count}, {batch_idx}/{loader_len}", end="")

    global_gt = gt
    print('\nSINGLE VOTING RESULTS\n')

    predictions = list(chain.from_iterable(predictions))
    print(classification_report(gt,predictions,target_names=dataset.classes_names))
    conf_matrix = confusion_matrix(gt.astype(int).tolist(), predictions)
    print(conf_matrix)

    # conf_dict = analyse_confusion_matrix(conf_matrix)
    # accuracy = accuracy_score(gt.astype(int).tolist(), predictions)
    # for i in range(dataset.classes_count):
    #     print(f" >stats for: {dataset.class_inv_dict[i]} - F1: {conf_dict['F1'][i]:.3f}, TPR: {conf_dict['TPR'][i]:.3f}, PPV: {conf_dict['PPV'][i]:.3f},")
    [predictions_all[idx].append(pred) for idx, pred in enumerate(predictions)]


print(f"\nFINAL VOTING BASED ON {args.votes_count} VOTES")
print(f"Used model:{args.trained_model}")
final_pred = []
votes_all = []
for predict in predictions_all:
        votes = Counter(predict).most_common(n=1)
        votes_all.append(votes)
        pred = votes[0][0]
        final_pred.append(pred)


print(classification_report(global_gt,final_pred,target_names=dataset.classes_names, digits=3))
conf_matrix = confusion_matrix(global_gt.astype(int).tolist(), final_pred)
print(conf_matrix)

# create csv file with results of inference
list_preds = []
for id, (pred,gt,votes) in enumerate(zip(final_pred, global_gt.astype(int).tolist(), predictions_all)):
    result = []
    result.append(dataset.img_paths[id])
    result.append(dataset.class_inv_dict[pred])
    result.append(dataset.class_inv_dict[gt])
    if gt == pred:
        result.append("true")
    else:
        result.append("false")
    count = Counter(votes)
    result.extend([count[0], count[1], count[2]])
    list_preds.append(result)

inference_data = pd.DataFrame(list_preds, columns=['filepath','prediction', 'ground_truth','is_correct', 'norm_v', 'pneu_v', 'cov_v'])
inference_data.to_csv('inference_data.csv')
