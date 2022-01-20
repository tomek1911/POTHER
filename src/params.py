import argparse

parser = argparse.ArgumentParser()

#PATHS
parser.add_argument('--dataset', type=str, default='covidx', choices={'darwin', 'covidx', 'wang', 'bimcv'}) 
parser.add_argument('--dir', type=str, default='models')
parser.add_argument('--classes_num', type=int, default=3)
parser.add_argument('--pretrained_bb_path', type=str, default='pretrained_models/resnet50_nih_8_epochs.pth')

#DARWIND PATHS
parser.add_argument('--data_path', type = str, default = 'v7darwin_1024', help = 'This is the path of the training data')
parser.add_argument('--dataset_csv', type = str, default = 'csv_files/v7darwin_dataset', help = 'This is the path of the training data')
parser.add_argument('--bb_file', type = str, default='csv_files/darwin_masks_bounding_boxes.csv')

#COVIDX PATHS
parser.add_argument('--data_path_covidx', type = str, default = '../CovidXBenchmark/benchmark_data', help = 'This is the path of the training data')
parser.add_argument('--dataset_csv_covidx', type = str, default = '../CovidXBenchmark/benchmark_data/covidx_mask', help = 'This is the path of the training data')
parser.add_argument('--dataset_source_task_path', type = str, default = '../CovidXBenchmark/benchmark_data/csv_dataset_task_source/covidx_mask')
parser.add_argument('--equ_dataset', action='store_true')
parser.add_argument('--validation', action='store_true')

#WANG_BENCHMARK PATHS
parser.add_argument('--data_path_wang', type = str, default = '../CovidXBenchmark/wang_benchmark', help = 'This is the path of the training data')
parser.add_argument('--dataset_csv_wang', type = str, default = '../CovidXBenchmark/wang_benchmark/wang_masked', help = 'This is the path of the training data')

#BIMCV PATHS
parser.add_argument('--data_path_bimcv', type = str, default = '../CovidXBenchmark/BIMCV', help = 'This is the path of the training data')
parser.add_argument('--dataset_csv_bimcv', type = str, default = '../CovidXBenchmark/BIMCV/bimcv', help = 'This is the path of the training data')

#SANITY TESTS
parser.add_argument('--nolungs', action='store_true')
parser.add_argument('--onlylungs', action='store_true')
parser.add_argument('--masklungs', action='store_true')


#HYPERPARAMS
parser.add_argument('-m','--model', type = str, default='resnet18', choices={'resnet18','resnet34', 'resnet50'})
parser.add_argument('--ptr', action = 'store_true', help = 'will use pretrained model - imagenet')
parser.add_argument('--bs', type = int, default = 8, help = 'batch size')
parser.add_argument('--lr', type = float, default = 1e-5, help = 'Learning Rate for the optimizer')
parser.add_argument('-e', '--epochs', type=int, default=3, help='Number of epochs of training')
parser.add_argument('-g', '--gamma', type=float, default=2.0, help='Gamma param for focal loss')
parser.add_argument('--wd', type = float, default = 1e-4, help = 'Weight decay parameter')
parser.add_argument('--ens_beta', type = float, default = 0.998, help = 'param for effecitve samples number tail lenght')
parser.add_argument('--input_size', type = int, default = 224, help = 'Image input size')
parser.add_argument('--sched_gamma', type = float, default = 0.1, help = 'gamma parameter for multistep-scheduler')
parser.add_argument('--sched_steps', nargs="+", type=int, default = [2,30,50])

#MULTITASK
parser.add_argument('--T1', action='store_false') # remove classification task
parser.add_argument('--T2', action='store_true') # add segmentation task
parser.add_argument('--T3', action='store_true') # add reconstruction task
parser.add_argument('--weighted_multitask_loss', action='store_true')
parser.add_argument("--loss_seg_mult", type=int, default=5)

#PATCH LEARNING
parser.add_argument('--use_patch', action='store_true')
parser.add_argument('--patch_base', type=int, default=112)
parser.add_argument('--patch_base_mult', type=float, default=2.0)
parser.add_argument('--patch_preview', action='store_true') 
parser.add_argument('--visualise', action='store_true') # only for visualisation - gradcam
parser.add_argument('--random_patch_size', action='store_true')
parser.add_argument('--simple_area_draw', action='store_true')
parser.add_argument('--crop_ratio', type=float, default=0.75)
parser.add_argument('--non_zero_mask_draw', action='store_true')
parser.add_argument('--use_reduced_draw', action='store_true')
parser.add_argument('--non_random_patch', action='store_true')

#UNET
parser.add_argument('--use_attention', action='store_true')
parser.add_argument('--experimental_head', action='store_true')
parser.add_argument('--norm_features', action='store_true')

#INFERENCE
parser.add_argument('--votes_count', type=int, default=25)
parser.add_argument('--trained_model', type=str, default='19-12-2021_03-31-26/resnet18_ep_20.pth', help='provide path to the model that will be used to continue training')

#FLAGS & UTILS
parser.add_argument('--mode', type = str, default = 'titan', choices={'debug', 'titan'})
parser.add_argument('-t','--test', action = 'store_true') # run inference
parser.add_argument('--verbosity', type=int, default=1, choices={0,1,2}, help = '0 - quiet, 1 - basic log, 2 - detailed info')
parser.add_argument('--comet', action = 'store_true')
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--log_img', action = 'store_true')
parser.add_argument('--log_img_freq', type=int, default=100)
parser.add_argument('--save', action = 'store_true')
parser.add_argument('--save_interval', type = int, default=25, help='epoch count interval')

#DOMAIN ADAPTATION
parser.add_argument('--use_ptr_bb', action = 'store_true')
parser.add_argument('--soft_tl', action = 'store_true', help = 'soft transfer learning - unfreeze layers starting from head')

#TRAINING
parser.add_argument('--tags', type = str, default = '')
parser.add_argument('--stage', type = int, default = 1, help = 'Stage, it decides which layers of the Neural Net to train')
parser.add_argument('--device', type = str, default = "1", choices = {"0", "1", "-1"}, help = 'gpu ID, -1 is cpu')
parser.add_argument('--limited', type = int, default = -1, help = 'Limited amount of data for fast test')
parser.add_argument('--loss_func', type = str, default = 'CE', choices = {'CE', 'BCE', 'FocalLoss'}, help = 'loss function')
parser.add_argument('--opti', type = str, default = "Adam", choices = {"Adam", "AdamW"}, help = 'Weight decay parameter')
parser.add_argument('--weights', type = str, default='none', choices = {"none", "inverse_freq", "ens"})
parser.add_argument('--augmentation', type = str, default = "albumentations", choices = {"torchvision", "albumentations", "global_albu", "strong_augment"}, help = 'Version of augmentations')
parser.add_argument('--workers', type = int, default = 8, help = 'number of workers - for dataloader')
parser.add_argument('--seed', type = int, default = 0, help = 'set seed for results reproducibility')
parser.add_argument('--det', action='store_true', help='will use deterministic cuda behaviour')
parser.add_argument('--sched', action = 'store_true', help = 'will use scheduler')
parser.add_argument('--freeze_layers', action = 'store_true')
parser.add_argument('--freeze_features', action = 'store_true')
parser.add_argument('--diff_lr', action = 'store_true')
parser.add_argument('--trained_layers', nargs="+", type=str, default = ['layer2','layer3','layer4','fc'])
parser.add_argument('--lr_mods', nargs="+", type=int, default = [1,2,10,20,100])
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--prev_epochs', type=int, default=0)

args = parser.parse_args()
