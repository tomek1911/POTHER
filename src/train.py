import torch
from torch._C import device
import torch.nn.functional as f
import numpy as np
from torch.utils.data import dataset
import torch.optim as optim
from src.utils import getImageFromBatchIdx, showImageFromBatchIdx, analyse_confusion_matrix, tensor2PIL, inv_norm_1CH, tensor_img2np, save_model, freeze_layers, getTrainable, onehotEncodeIntegerLabels
from skimage.color import label2rgb
from itertools import chain
from PIL import Image
import cv2
from datetime import datetime

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def train_schedule(epoch, dataset, dataset_val):

    #training
    if epoch == 0:
        dataset.patch = False
    if epoch == 25:
        dataset.patch = True
        dataset.random_patch_size = False
        dataset.patch_base = 112 # 224x224
    if epoch == 50:
        dataset.patch = True
        dataset.random_patch_size = True
        dataset.patch_base = 56 # 112x112 - 448x448
    #validation
    if epoch == 0:
        dataset_val.patch = False
    if epoch == 25:
        dataset_val.patch = True
        dataset_val.random_patch_size = False
        dataset_val.patch_base = 112 # 224x224
    if epoch == 50:
        dataset_val.patch = True
        dataset_val.random_patch_size = True
        dataset_val.patch_base = 56 # 112x112 - 448x448

def train_stage(epoch, args, model, optimizer):
    
    if epoch == 0: #epoch name:1
        print ("Transfer learning, steps in epochs: 2, 4, 6, 11")
        print("Freezing all layers without head")
        freeze_layers(model, ['layer1', 'layer2', 'layer3', 'layer4'])
        print(f"Trainable parameters: {getTrainable(model)}")
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return optim.RAdam(params=model_parameters, lr=args.lr, weight_decay=args.wd)
    elif epoch == 1: #epoch name:2
        print("Unfreeze layer4")
        freeze_layers(model, ['layer1', 'layer2', 'layer3'])
        print(f"Trainable parameters: {getTrainable(model)}")
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return optim.RAdam(params=model_parameters, lr=args.lr, weight_decay=args.wd)
    elif epoch == 3: #epoch name:4
        print("Unfreeze layer3")
        freeze_layers(model, ['layer1', 'layer2'])
        print(f"Trainable parameters: {getTrainable(model)}")
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return optim.RAdam(params=model_parameters, lr=args.lr, weight_decay=args.wd)
    elif epoch == 5: #epoch name:6
        print("Unfreeze layer2")
        freeze_layers(model, ['layer1'])
        print(f"Trainable parameters: {getTrainable(model)}")
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print(f"Trainable parameters: {getTrainable(model)}")
        return optim.RAdam(params=model_parameters, lr=args.lr, weight_decay=args.wd)
    elif epoch == 10: #epoch name:11
        print("Unfreeze layer1")
        freeze_layers(model, ['none'])
        print(f"Trainable parameters: {getTrainable(model)}")
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return optim.RAdam(params=model_parameters, lr=args.lr, weight_decay=args.wd)
    return optimizer
      
def train(args, model, optimizer, scheduler, dataset, dataset_val, train_loader, val_loader, conf, tasks,
          loss_class_fn, loss_seg_fn, loss_rec_fn, dice, exp=None):
    task_count = 0
    if tasks['T1']:
        task_count+=1
    if tasks['T2']:
        task_count+=1
    if tasks['T3']:
        task_count+=1

    print_once = True
    dataset_len = len(dataset)
    dataset_classes_count = dataset.classes_count
    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    #TRAINING
    print("\n\n TRAINING \n")
    for epoch in range(args.prev_epochs, args.epochs+args.prev_epochs):
        
        print(f"epoch: {epoch+1}/{args.epochs+args.prev_epochs}\n")

        # train_schedule(epoch, dataset, dataset_val)
        if args.soft_tl:
            optimizer = train_stage(epoch, args, model, optimizer)

        model.train()
        torch.set_grad_enabled(True)

        running_train_loss = 0
        running_train_class_loss = 0
        running_train_segm_loss = 0
        running_train_rec_loss = 0
        dice_coefficient_train = 0
        predictions = []

        #probabilities for metrics
        probs = np.zeros((dataset_len, dataset_classes_count), dtype = np.float32)             
        gt    = np.zeros(dataset_len, dtype = np.float32)
        k=0

        loss_classification, loss_segmentation, loss_reconstruction = 0, 0, 0

        for batch_idx, (img, mask, label) in enumerate(train_loader):

            img = img.to(conf.device, dtype=torch.float32)
            mask = mask.to(conf.device, dtype=torch.float32)
            target = label.to(conf.device, dtype=torch.long)
            # if print_once:
            #     print("Init Debug Log:")
            #     print(f" >Batch: {batch_idx+1}/{len(train_loader)}, labels: {label.tolist()}, image: {(batch_idx+1)*args.bs}/{dataset_len}.")
            #     print(f" >Image - type: {img.type()}, device: {img.device.type}, shape: {img.shape}.")
            #     print(f" >Mask - type: {mask.type()}, device: {mask.device.type}, shape: {mask.shape}.")
            #     print("\n")
            #     print_once = False

            optimizer.zero_grad()
            x_class, x_seg, x_res = model(img)
            if args.classes_num == 2:
                target = label.to(conf.device, dtype=torch.float32)
                loss_classification = loss_class_fn(x_class, target.unsqueeze(1))
            else:
                target = label.to(conf.device, dtype=torch.long)
                loss_classification = loss_class_fn(x_class, target)
            if tasks['T2']:
                segm_activation = torch.sigmoid(x_seg)
                loss_segmentation = args.loss_seg_mult * loss_seg_fn(segm_activation, mask)
            if tasks['T3']:
                rec_activation = torch.sigmoid(x_res)
                rec_target = img * mask
                loss_reconstruction = loss_rec_fn(rec_activation, rec_target)

            # focal_loss = loss_mask_fn2(segm_activation, mask)
            # dice_loss_log = torch.log(dice_loss)
            # loss_mask = 7.0 * focal_loss - dice_loss_log 

            if args.weighted_multitask_loss:       
                if task_count == 2:
                    weight = f.softmax(torch.randn(2), dim=-1).to(conf.device)
                    loss = (loss_classification * weight[0] + loss_segmentation * weight[1])*2
                elif task_count ==3:
                    weight = f.softmax(torch.randn(3), dim=-1).to(conf.device)
                    loss = (loss_classification * weight[0] + loss_segmentation * weight[1] + loss_reconstruction * weight[2])*3
            else:
                if task_count == 1:
                    loss = loss_classification
                elif task_count ==2:
                    loss = loss_classification + loss_segmentation
                elif task_count==3:
                    loss = loss_classification + loss_segmentation + loss_reconstruction
            
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()
                print(f"Scheduler learning rate {scheduler.get_lr()}")

            running_train_loss += loss.item()
            running_train_class_loss += loss_classification.item()
            if tasks['T2']:
                running_train_segm_loss += loss_segmentation.item()
                dice_coefficient_train += dice(segm_activation, mask).item()
            if tasks['T3']:            
                running_train_rec_loss += loss_reconstruction.item()
            

            # store predictions for metric evaluation 
            if args.classes_num ==2:
                pred = (f.sigmoid(x_class).detach().cpu().numpy()>0.5).astype(np.int)
                predictions.append(pred)
            else:
                predictions.append(x_class.max(1).indices.tolist())

            probs[k: k + x_class.shape[0], :] = f.softmax(x_class, dim = 1).detach().cpu()
            gt[   k: k + x_class.shape[0]] = target.detach().cpu()
            k += x_class.shape[0]

            if (batch_idx) % args.log_freq == 0:
                mean_loss = running_train_loss / ((batch_idx+1))
                mean_class_loss = running_train_class_loss / ((batch_idx+1))
                mean_segm_loss = running_train_segm_loss / ((batch_idx+1))
                mean_rec_loss = running_train_rec_loss / ((batch_idx+1))
                dice_coefficient_mean = dice_coefficient_train / ((batch_idx+1))
                print(f'batch: {batch_idx+1}/{len(train_loader)} - mean_loss: {mean_loss:.4f}, class_loss: {mean_class_loss:.4f}, segm_loss: {mean_segm_loss:.4f}, rec_loss: {mean_rec_loss:.4f}, dice_coeff: {dice_coefficient_mean:.4f} ')
                step = (batch_idx+1)+(epoch*len(train_loader))
                if exp: 
                    exp.log_metric("train_loss_avg_step", mean_loss, step=step)
                    exp.log_metric("train_class_loss_avg_step", mean_class_loss, step=step)
                    exp.log_metric("train_segm_loss_avg_step", mean_segm_loss, step=step)
                    exp.log_metric("train_dice_step", dice_coefficient_mean, step=step)
                    if args.log_img and batch_idx % args.log_img_freq == 0:
                        lung_log = tensor_img2np(img, denorm=True)
                        if tasks['T2']:
                            segm_log = tensor_img2np(segm_activation)
                        else:
                            segm_log = np.zeros((args.input_size, args.input_size), dtype=np.uint8)
                        if tasks['T3']:
                            rec_log = tensor_img2np(rec_activation)
                        else:
                            rec_log = np.zeros((args.input_size, args.input_size), dtype=np.uint8)
                        mask_log = tensor_img2np(mask)
                        rec_target_log = (lung_log*(mask_log/255)).astype(np.uint8)
                        log_image = cv2.hconcat([lung_log, rec_log, rec_target_log, segm_log, mask_log])
                        exp.log_image(log_image, name=f"{epoch+1:03d}_{batch_idx+1:04d}_img_rec_recT_seg_mask", image_channels='first')
                        # exp.log_image(rec_log, name=f"{epoch+1:03d}_{batch_idx+1:04d}_reconstruction", image_channels='first')
                        ##overlay mask on original cxr
                        # segm_log_bin = segm_log * (segm_log > 128)
                        # overlay_mask = np.uint8(label2rgb(segm_log_bin, lung_log, kind='overlay', bg_label=0, colors=[(0.2,0.8,0.4)], image_alpha=0.9)*255)
                        # exp.log_image(overlay_mask, name=f"{epoch}_{batch_idx+1}_overlay", image_channels='last')

                    
        print('\nTRAINING RESULTS\n')

        #full report
        predictions = list(chain.from_iterable(predictions))
        print(classification_report(gt,predictions,target_names=dataset.classes_names, digits=3))
        #confusion matrix
        conf_matrix = confusion_matrix(gt.astype(int).tolist(), predictions)
        print(conf_matrix)
        #metrics dictionary
        conf_dict = analyse_confusion_matrix(conf_matrix)
        accuracy = accuracy_score(gt.astype(int).tolist(), predictions)

        # for i in range(dataset.classes_count):
        #     print(f" >stats for: {dataset.class_inv_dict[i]} - F1: {conf_dict['F1'][i]:.4f}, TPR: {conf_dict['TPR'][i]:.4f}, PPV: {conf_dict['PPV'][i]:.4f},")

        if exp:
            exp.log_confusion_matrix(matrix = conf_matrix, labels = ['normal', 'pneumonia', 'covid-19'], title=f"Training Confusion Matrix,  Epoch: {epoch + 1:03d}", file_name=f"confusion-matrix-train-{epoch + 1:03d}.json")
            exp.log_metric("train_loss_epoch", running_train_loss / len(train_loader), step=(epoch+1))
            exp.log_metric("train_acc_epoch", accuracy, step=(epoch+1))
            if args.classes_num == 3:
                exp.log_metric("train_F1_covid_epoch", conf_dict['F1'][2], step=(epoch+1)) # 2 is covid-19
                exp.log_metric("train_PPV_covid_epoch", conf_dict['PPV'][2], step=(epoch+1))
                exp.log_metric("train_TPR_covid_epoch",conf_dict['TPR'][2], step=(epoch+1)) 
            exp.log_metric("train_dice_coeff_epoch", dice_coefficient_train/len(train_loader), step=(epoch+1))
            exp.log_metric("train_mean_F1_epoch", conf_dict['F1'].mean(), step=(epoch+1))  
            exp.log_metric("train_mean_PPV_epoch", conf_dict['PPV'].mean(), step=(epoch+1))  
            exp.log_metric("train_mean_TPR_epoch", conf_dict['TPR'].mean(), step=(epoch+1))    
   
        
        #################################################################################################################
        #VALIDATION
        print("\nVALIDATION")
        
        model.eval()
        torch.set_grad_enabled(False)
        dataset_val_len = len(dataset_val)
        #metrics containers
        probs = np.zeros((dataset_val_len, dataset_classes_count), dtype = np.float32)             
        gt    = np.zeros(dataset_val_len, dtype = np.float32)
        k=0
        running_val_loss = 0
        running_val_class_loss = 0
        running_val_segm_loss = 0
        dice_coefficient_val = 0
        predictions = []

        for batch_idx, (img, mask, label) in enumerate(val_loader):

            img = img.to(conf.device, dtype=torch.float32)
            mask = mask.to(conf.device, dtype=torch.float32)
            # target = label.to(conf.device, dtype=torch.long)

            x_class, x_seg, x_res = model(img)
            if args.classes_num == 2:
                target = label.to(conf.device, dtype=torch.float32)
                loss_classification = loss_class_fn(x_class, target.unsqueeze(1))
            else:
                target = label.to(conf.device, dtype=torch.long)
                loss_classification = loss_class_fn(x_class, target)
            if tasks['T2']:
                segm_activation = torch.sigmoid(x_seg)
                loss_segmentation = args.loss_seg_mult * loss_seg_fn(segm_activation, mask)
            if tasks['T3']:
                rec_activation = torch.sigmoid(x_res)
                loss_reconstruction = loss_rec_fn(rec_activation, img)
        
            loss = loss_classification + loss_segmentation # + loss_reconstruction

            running_val_loss += loss.item()
            running_val_class_loss += loss_classification.item()
            if tasks['T2']:
                running_val_segm_loss += loss_segmentation.item()
                dice_coefficient_val += dice(segm_activation, mask).item()

            # store predictions for metric evaluation 
            if args.classes_num ==2:
                pred = (f.sigmoid(x_class).detach().cpu().numpy()>0.5).astype(np.int)
                predictions.append(pred)
            else:
                predictions.append(x_class.max(1).indices.tolist())
            probs[k: k + x_class.shape[0], :] = f.softmax(x_class, dim = 1).detach().cpu()
            gt[   k: k + x_class.shape[0]] = target.detach().cpu()
            k += x_class.shape[0]
            print(f"\rValidation in progress: {batch_idx}/{len(val_loader)}.", end="")

            # if (batch_idx) % args.log_freq == 0:
            #     mean_loss = running_val_loss / ((batch_idx+1))
            #     mean_class_loss = running_val_class_loss / ((batch_idx+1))
            #     mean_segm_loss = running_val_segm_loss / ((batch_idx+1))
            #     step = (batch_idx+1)+(epoch*len(train_loader))
            #     if exp: 
            #         exp.log_metric("val_loss_avg_step", mean_loss, step=step)
            #         exp.log_metric("val_class_loss_avg_step", mean_class_loss, step=step)
            #         exp.log_metric("val_segm_loss_avg_step", mean_segm_loss, step=step)
            #         exp.log_metric("val_dice_step", dice_coefficient, step=step)
        
        print('\nVALIDATION RESULTS\n')

        #full report
        predictions = list(chain.from_iterable(predictions))
        print(classification_report(gt,predictions,target_names=dataset_val.classes_names, digits=3))
        #confusion matrix
        gt_list = gt.astype(int).tolist()
        conf_matrix = confusion_matrix(gt_list, predictions)
        print(conf_matrix)
        #metrics dictionary
        conf_dict = analyse_confusion_matrix(conf_matrix)
        accuracy = accuracy_score(gt_list, predictions)

        # for i in range(dataset_val.classes_count):
        #     print(f" >stats for: {dataset_val.class_inv_dict[i]} - F1: {conf_dict['F1'][i]:.4f}, TPR: {conf_dict['TPR'][i]:.4f}, PPV: {conf_dict['PPV'][i]:.4f},")

        if exp:
            exp.log_confusion_matrix(matrix = conf_matrix, labels = ['normal', 'pneumonia', 'covid-19'], title=f"Validation Confusion Matrix,  Epoch: {epoch + 1:03d}", file_name=f"confusion-matrix-valid-{epoch + 1:03d}.json")
            exp.log_metric("val_loss_epoch", running_val_loss / len(val_loader), step=(epoch+1))
            exp.log_metric("val_acc_epoch", accuracy, step=(epoch+1))
            if args.classes_num == 3:
                exp.log_metric("val_F1_covid_epoch", conf_dict['F1'][dataset_val.class_dict['COVID-19']], step=(epoch+1))
                exp.log_metric("val_PPV_covid_epoch", conf_dict['PPV'][dataset_val.class_dict['COVID-19']], step=(epoch+1))
                exp.log_metric("val_TPR_covid_epoch",conf_dict['TPR'][dataset_val.class_dict['COVID-19']], step=(epoch+1))  
            exp.log_metric("val_dice_coeff_epoch", dice_coefficient_val/len(val_loader), step=(epoch+1))  
            exp.log_metric("val_mean_F1_epoch", conf_dict['F1'].mean(), step=(epoch+1))  
            exp.log_metric("val_mean_PPV_epoch", conf_dict['PPV'].mean(), step=(epoch+1))  
            exp.log_metric("val_mean_TPR_epoch", conf_dict['TPR'].mean(), step=(epoch+1))  

        if (epoch+1) % args.save_interval == 0 and args.save:
            save_model(args, model, optimizer, (epoch+1), date_str, conf_dict)