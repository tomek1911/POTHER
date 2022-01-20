import os
import albumentations as albu
import numpy as np
import torch
import json 
from scipy import sparse
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from torchvision.transforms.transforms import Grayscale

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])
norm_albu = albu.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
norm_albu_1CH = albu.Normalize(mean=[0.5], std=[0.5])
inv_norm_1CH = transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5])

# GRAYSCALE
def tensor_img2np(tensor, denorm=False):
    tensor = tensor.cpu().detach()
    if denorm:
        out = tensor2PIL(tensor[0], denorm=True, denorm_transform=inv_norm_1CH).convert('L')
    else:
        out = tensor2PIL(tensor[0], denorm=False).convert('L')
    return np.array(out)
                
def np2PIL(img, denorm=True):
    if denorm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            inv_normalize,
            transforms.ToPILImage()
        ])
        return transform(img)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])
        return transform(img)


def tensor2PIL(img, denorm=True, denorm_transform=inv_normalize):
    if denorm:
        transform = transforms.Compose([
            denorm_transform,
            transforms.ToPILImage()
        ])
        return transform(img)
    else:
        transform = transforms.ToPILImage()
        return transform(img)

def tensor2PIL_1ch(img, denorm=True, denorm_transform=inv_norm_1CH):
    if denorm:
        transform = transforms.Compose([
            denorm_transform,
            transforms.ToPILImage()
        ])
        return transform(img)
    else:
        transform = transforms.ToPILImage()
        return transform(img)

def getImageFromBatchIdx(batch, denorm=True, idx=0):
    transform = transforms.ToPILImage()
    img1 = batch[idx]
    img2 = None
    if denorm:
        img2 = transform(inv_normalize(img1))
    else:
        img2 = transform(img1)
    return img2

def showImageFromBatchIdx(batch, denorm=True, idx=0, grayscale = False):
    transform = transforms.ToPILImage()
    img1 = batch[idx]
    img2 = None
    if denorm:
        img2 = transform(inv_normalize(img1))
    else:
        if grayscale:
            img2 = transform(img1).convert('L')
        else:
            img2 = transform(img1)
    img2.show()

#ONE HOT 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def integerEncodeLabels(labels : np.array):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    enc_dict = {k: i for (i,k) in enumerate (label_encoder.classes_)}
    return integer_encoded, enc_dict

def onehotEncodeIntegerLabels(int_labels : np.array, cat = 'auto'):
    onehot_encoder = OneHotEncoder(sparse=False, dtype=float, categories=cat)
    int_labels = int_labels.reshape(len(int_labels), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_labels)
    return onehot_encoded, onehot_encoder.categories_

def cat21hot(labels : np.array):
    int_enc, int_enc_dict = integerEncodeLabels(labels)
    onehot_enc, one_hot_cat = onehotEncodeIntegerLabels(int_enc)
    return onehot_enc, one_hot_cat, int_enc_dict

#Debug
# categorical_labels = ['covid', 'normal', 'normal','pneumonia', 'pneumonia', 'covid', 'normal', 'pneumonia']
# int_labels, int_enc_dict = integerEncodeLabels(categorical_labels)
# onehot_labels, onehot_enc_cat = onehotEncodeIntegerLabels(int_labels)

def print_list_float(list, prec=3):
    return [float(f'{item:.{prec}f}') for item in list]

#METRICS

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


def plot_pr_curve(gt, probs, class_name, mode='train'):
    plt.figure()
    precision, recall, _ = precision_recall_curve(gt, probs)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    path = os.path.join("plots", mode, f"pr_curve_{class_name}.png")
    plt.savefig(path)
    plt.clf()


def plot_roc_curve(gt, probs, class_name, mode='train'):
    plt.figure()
    fpr, tpr, _ = roc_curve(gt, probs)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    path = os.path.join("plots", mode, f"roc_curve_{class_name}.png")
    plt.savefig(path)
    plt.clf()

def get_roc_auc_score(gt, probs):

    class_roc_auc_list = []    
    for i in range(gt.shape[1]):
        class_roc_auc_list.append(roc_auc_score(gt[:, i], probs[:, i]))
    return np.mean(np.array(class_roc_auc_list)), class_roc_auc_list

def analyse_confusion_matrix(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    # TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    # NPV = TN/(TN+FN)
    # Fall out or false positive rate
    #FPR = FP/(FP+TN)
    # False negative rate
    #NR = FN/(TP+FN)
    # False discovery rate
    #FDR = FP/(TP+FP)

    ACC = (TP+TN)/(TP+FP+FN+TN)

    F1 = (2*TP)/(2*TP + FP + FN)
    return {"TPR":TPR,"PPV":PPV,"F1":F1,"ACC":ACC}

def save_hyperparams(args, params_dict, date_str):
    #dict to string
    params_dict = {k: str(v) for (k, v) in params_dict.items()}
    path = os.path.join(args.dir, date_str, 'hyperparameters.json')
    with open(path, "a+") as outfile:
            json.dump(params_dict, outfile, indent=4)
    outfile.close()
    print(f'Hyperarameters {path} saved.')

def save_model(args, model, optim, epoch, date_str, score_dict, hyperparams = None):

    save_name = f'{args.model}_ep_{str(epoch).zfill(2)}'
    save_folder = os.path.join(args.dir, date_str)
    save_path = os.path.join(save_folder, save_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(save_path+'_param'+'.json', "a+") as outfile:
        json.dump(vars(args), outfile, indent=4)
    outfile.close()
    
    torch.save({
        'parameters': vars(args),
        'score_dict': score_dict,
        'model': model,
        # 'model_state': model.state_dict(),
        # 'optim_state': optim.state_dict()
    }, save_path+'.pth')

    # save scores to json
    score_dict = {k: str(print_list_float(v)) for (k, v) in score_dict.items()}
    with open(save_path+'.json', "a+") as outfile:
        json.dump(score_dict, outfile, indent=4)
    outfile.close()

    print(f'Checkpoint and score {save_name} saved.')


def hyperparams2dict(args, extra, weights=""):
    hyper_dict = {
        "model_architecture": args.model,
        "pretrained model:": args.ptr,
        "loss_fn:": args.loss_func,
        "epochs_count:": args.epochs,
        "batch_size:": args.bs,
        "input size:": args.input_size,
        "base lr:": args.lr,
        "augmentations:": args.augmentation,
        "optimizer:": args.opti,
        "weight decay:": args.wd,
        "weights mode": args.weights,
        "weights values": weights
    }
    #add new optional hyperparams
    hyper_dict.update(extra)

    return hyper_dict

def freeze_layers(model, layers_list):
    for name, param in model.named_parameters(): 
        if any(layer in name for layer in layers_list):
            param.requires_grad = False 
        else:
            param.requires_grad = True

def getTrainable(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


