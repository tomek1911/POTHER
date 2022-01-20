import glob
import cv2 as cv
import numpy as np
import argparse
import os
import math
import csv
import time
import numpy as np
from comet_ml import Experiment

parser = argparse.ArgumentParser(description='filtration of segmented masks of lungs')

parser.add_argument('--input_dir', type = str, default = 'data/nih/masks/', help = 'This is the dir for the original masks')
parser.add_argument('--output', type = str, default = 'data/nih', help = 'This is the path of the output')
parser.add_argument('--output_dir', type = str, default = '../data/nih/masks_filtered/', help = 'This is the path of the output filtered masks')
parser.add_argument('--output_dir_interpolated', type = str, default = '../data/nih/masks_filtered_interpolated/', help = 'This is the path of the output filtered and interpolated masks')
parser.add_argument('--size', type = int, default = 512, help = 'Size of the output image')
parser.add_argument('--first_img', type = int, default = 0, help = 'First image from the list to process.')
parser.add_argument('--batch', type = int, default = -1, help = 'Amount of images in the folder to filter - for -1 will run for all')
parser.add_argument('--compression', type = int, default = 3, choices=[0,1,2,3,4,5,6,7,8,9], help = 'Compression value - 0: large file - fast, 9: small file - slow.')
parser.add_argument('--copy_rejected', action = 'store_true', help = 'will copy rejected original images to folders')
parser.add_argument('--rejected_to_csv', action = 'store_true', help = 'will create csv file with rejected images paths')
parser.add_argument('--file', action = 'store_true', help = 'will process only one image')
parser.add_argument('--file_dir', type = str, default = '', help = 'This is the dir of one image to process')
parser.add_argument('--log_images', action = 'store_true', help = 'will log all masks to comet experiment')
parser.add_argument('--rejection_conditions', nargs="+", type=str, default = ['disprop','solid','line'], choices=['disprop','solid','line'], help = "choose conditions on which you want to reject masks")
parser.add_argument('--bbox_pad', type=float, default=0.05, help = "padding percent around lungs bounding box")
parser.add_argument('--save', action = 'store_true', help = 'will save filtered masks pngs to folder')
parser.add_argument('--save_bboxes', action = 'store_true', help = 'will save bboxes to csv file')
args = parser.parse_args()

mask_files = glob.glob(args.input_dir + "/*.png", recursive=True)
mask_files.sort()

print(f"Found {len(mask_files)} images.")

mask_files = mask_files[args.first_img:]

if args.batch != -1:
    mask_files = mask_files[:args.batch]

print(f"About to filter {len(mask_files)} images.")
print("Filtration started: ")
class SegmentationMasksFilter():

    rejected_line_count = 0
    rejected_solidity_count = 0
    rejected_paths = []
    bounding_boxes = []
    ratios = []
    one_contour = 0
    two_contours = 0
    more_contours = 0
    high_disproportion = 0
    processed_images = 0
    current_path = ""
    reject_current_contour = False

    is_solid_reject = False
    is_line_reject = False
    is_disprop_reject = False    

    def __init__(self, paths, conditions) -> None:
        self.paths = paths
        self.images_count = len(self.paths)

        if 'solid' in conditions:
            self.is_solid_reject = True
        if 'line' in conditions:
            self.is_line_reject = True
        if 'disprop' in conditions:
            self.is_disprop_reject = True        
        
    def interpolate_mask(self, img, kernel_base = 4, upscaling_size = 1024, target_size = 512):

        kernel = math.ceil(img.shape[0] / 128.0) * kernel_base - 1   
        # img_resized = cv.resize(img,(upscaling_size, upscaling_size), interpolation=cv.INTER_NEAREST)
        # img_resized = cv.resize(img,(target_size, target_size), interpolation=cv.INTER_NEAREST)
        img = cv.GaussianBlur(img, (kernel,kernel), 3)
        ret, img = cv.threshold(img,2,255,cv.THRESH_BINARY)
        img = cv.morphologyEx(img,cv.MORPH_ERODE,np.ones((3,3), dtype=np.uint8), iterations=3)    
        return img

    def filterContours(self, contours, shape):
        # one_contour = 0
        # two_contours = 0
        # more_contours = 0
        # high_disproportion = 0
        out_img = np.zeros(shape, np.uint8)

        if len(contours) == 1:   
            self.one_contour+=1  
            self.reject_current_contour = True
            # TODO decide what to do with 1 contour images
            # so far they are rejected

        elif len(contours) == 2:
            self.two_contours+=1
            area0 = cv.contourArea(contours[0])
            area1 = cv.contourArea(contours[1])
            max_area = max(area0, area1)
            min_area = min(area0, area1)
            if max_area / 2 < min_area:
                cv.fillPoly(out_img, pts = contours, color = 255)    
            else:
                if self.is_disprop_reject:
                    out_img = np.zeros(shape, np.uint8)
                    self.reject_current_contour = True
                    self.high_disproportion+=1    
                    self.rejected_paths.append([self.current_path, 'disprop'])
                else:
                    cv.fillPoly(out_img, pts = contours, color = 255)

        elif len(contours) > 2:
            
            self.more_contours+=1
            areas = []
            for cnt in contours:
                areas.append(cv.contourArea(cnt))    
            # indices of two largets areas
            ind = np.argpartition(areas, -2)[-2:]    
            
            area0 = cv.contourArea(contours[ind[0]])
            area1 = cv.contourArea(contours[ind[1]])
            max_area = max(area0, area1)
            min_area = min(area0, area1)        
            if max_area / 2 < min_area:
                fill = [contours[i] for i in ind]    
                cv.fillPoly(out_img, pts = fill, color = 255)            
            else:
                if self.is_disprop_reject:
                    out_img = np.zeros(shape, np.uint8)  
                    self.reject_current_contour = True              
                    self.high_disproportion+=1
                    self.rejected_paths.append([self.current_path, 'disprop'])
                else:
                    cv.fillPoly(out_img, pts = contours, color = 255)
                    
        return out_img

    def filter_contours(self, img):
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        img = self.filterContours(contours, img.shape)
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return img, contours

    def solidity_calc(self, cnt):    
        area = cv.contourArea(cnt)
        hull_area = cv.contourArea(cv.convexHull(cnt))
        return area / hull_area

    def is_contour_fixable(self, contour, margin=0.75):
        solidity = self.solidity_calc(contour)
        fixable = False
        if solidity > margin:            
            fixable = True    
        return fixable

    def fit_line_is_reject(self, contours, left, right, shape = (224,224)):
        reject_contour = False
        rows,cols = shape[:2]
        img_cx, img_cy = int(rows/2), int(cols/2)

        for cnt in contours:
        
            #CENTER OF MASS - L or R lung contour
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            lung_flag = ''
            if cX <img_cx:
                lung_flag = 'l'           
            else:
                lung_flag = 'r'

            #FIT LINE        
            [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)   
            angle = math.atan(vy / vx) * 180 / math.pi    
            # cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
            if lung_flag == 'l':
                if not ((angle >= left[0] and angle <= left[1]) or (angle >= 85 and angle <= 90)):                    
                    reject_contour = True                  
            
            if lung_flag == 'r':
                if not ((angle >= right[0] and angle <= right[1]) or (angle >= -90 and angle <= -85)):     
                    reject_contour = True         

            return reject_contour  
    
    def filterOneFile(self, mfile):
            
        self.reject_current_contour = False
        self.current_path = mfile

        #Default OUTPUT is an empty image 
        output_img = np.zeros(args.size, dtype=np.uint8) 
        out_img_interp = np.zeros(args.size, dtype=np.uint8)  

        img = cv.imread(mfile, cv.IMREAD_GRAYSCALE)

        if args.log_images:
            self.experiment.log_image(img, name=f"{id:05}_nih_org_{os.path.basename(mfile)}", image_channels='first')

        img_filtered, contours = self.filter_contours(img)

        #After contour filtration OUTPUT is equal to filtration output
        output_img = img_filtered    

        if self.is_line_reject and not self.reject_current_contour:
            if self.fit_line_is_reject(contours, left=[-90, -50], right = [50, 90], shape = img_filtered.shape):
                
                self.reject_current_contour = True
                output_img = np.zeros(args.size, dtype=np.uint8) 
                self.rejected_line_count+=1  
                self.rejected_paths.append([mfile, 'line'])         
        
        if self.is_solid_reject and not self.reject_current_contour:
            
            solidities = []
            for cnt in contours:
                solidities.append(self.solidity_calc(cnt))                             

            if all([x > 0.8 for x in solidities]):
                pass
                # In case of success nothing has to be done                   
            else:
                self.reject_current_contour = True
                output_img = np.zeros(args.size, dtype=np.uint8) 
                self.rejected_solidity_count+=1 
                self.rejected_paths.append([mfile, 'solidity'])     

            if not self.reject_current_contour:
                output_img = cv.morphologyEx(output_img, cv.MORPH_OPEN, np.ones((3,3), dtype=np.uint8))
                output_img = cv.morphologyEx(output_img, cv.MORPH_DILATE, np.ones((3,3), dtype=np.uint8), iterations=2)
                out_img_interp = self.interpolate_mask(output_img)
            else:
                if args.log_images:
                    self.experiment.log_image(np.asarray(img), name=f"{id:05}_nih_rejected_{os.path.basename(mfile)}", image_channels='first')

        cv.imwrite(os.path.join(args.output_dir, mfile.split('/')[-1]), output_img, [cv.IMWRITE_PNG_COMPRESSION, args.compression]) 
        cv.imwrite(os.path.join(args.output_dir_interpolated, mfile.split('/')[-1]), out_img_interp, [cv.IMWRITE_PNG_COMPRESSION, args.compression])  
            
        print(f"Success, saved file: {mfile.split('/')[-1]}")

    def runFiltration(self):
        
        for id, mfile in enumerate(self.paths):
            
            self.reject_current_contour = False
            self.current_path = mfile

            #Target shape
            shape = (args.size, args.size)

            #Default OUTPUT is an empty image 
            output_img = np.zeros(shape, dtype=np.uint8) 
            out_img_interp = np.zeros(shape, dtype=np.uint8)  

            img = cv.imread(mfile, cv.IMREAD_GRAYSCALE)
            if img.shape != shape:
                img = cv.resize(img, shape, interpolation = cv.INTER_AREA)

            if args.log_images:
                self.experiment.log_image(img, name=f"{id:05}_nih_org_{os.path.basename(mfile)}", image_channels='first')

            img_filtered, contours = self.filter_contours(img)                        
            output_img = img_filtered    

            if self.is_line_reject and not self.reject_current_contour:
                if self.fit_line_is_reject(contours, left=[-90, -50], right = [50, 90], shape = img_filtered.shape):
                    
                    self.reject_current_contour = True
                    output_img = np.zeros(shape, dtype=np.uint8) 
                    self.rejected_line_count+=1  
                    self.rejected_paths.append([mfile, 'line'])         
            
            if self.is_solid_reject and not self.reject_current_contour:
                
                solidities = []
                for cnt in contours:
                    solidities.append(self.solidity_calc(cnt))                             

                if all([x > 0.8 for x in solidities]):
                    pass
                    # In case of success nothing has to be done                   
                else:
                    self.reject_current_contour = True
                    output_img = np.zeros(shape, dtype=np.uint8) 
                    self.rejected_solidity_count+=1 
                    self.rejected_paths.append([mfile, 'solidity'])     

                if not self.reject_current_contour:
                    output_img = cv.morphologyEx(output_img, cv.MORPH_OPEN, np.ones((3,3), dtype=np.uint8))
                    output_img = cv.morphologyEx(output_img, cv.MORPH_DILATE, np.ones((3,3), dtype=np.uint8), iterations=2)
                    out_img_interp = self.interpolate_mask(output_img)
                else:
                    if args.log_images:
                        self.experiment.log_image(np.asarray(img), name=f"{id:05}_nih_rejected_{os.path.basename(mfile)}", image_channels='first')

            # IMAGES ARE FILTERED now we can get bounding boxes            
            if not self.reject_current_contour and args.save_bboxes and len(contours)>0:

                # bounding box around both lungs
                rect = cv.boundingRect(np.concatenate(contours,axis=0))
                x_min = rect[0]
                y_min = rect[1]
                x_max = x_min + rect[2]
                y_max = y_min + rect[3]           

                #padding
                padding_vector = math.floor(args.size * args.bbox_pad)
                x_min -= padding_vector
                y_min -= padding_vector
                x_max += padding_vector
                y_max += padding_vector

                #create closes padding that fits inside image
                while (x_min < 0):
                    x_min+=1
                while (y_min < 0):
                    y_min+=1
                while (x_max > args.size):
                    x_max-=1
                while (y_max > args.size):
                    y_max-=1
                    

                #albumentations format for corners of bounding box
                bb = [x_min, y_min, x_max, y_max]    
                bb_normalized = [float(pos) / args.size for pos in bb]

                self.bounding_boxes.append([mfile] + bb_normalized + [args.size])
                
                ## debug - show bbox image
                # empty = np.zeros(shape, dtype=np.uint8) 
                # img_bbox = cv.rectangle(empty,(x_min,y_min), (x_max,y_max),color=255,thickness=1)
                # cv.imshow("bbox", img_bbox)
                # cv.waitKey(0)
                
                #debug - check difference between boxes and full image
                roi_area = (y_max - y_min) * (x_max - x_min)
                img_area = args.size * args.size
                ratio = float(roi_area)/img_area
                self.ratios.append(ratio)
            elif self.reject_current_contour and args.save_bboxes:
                self.bounding_boxes.append([mfile] + [-1.0,-1.0,-1.0,-1.0] + [args.size])       

            if args.save:                
                cv.imwrite(os.path.join(args.output_dir, mfile.split('/')[-1]), output_img, [cv.IMWRITE_PNG_COMPRESSION, args.compression]) 
                cv.imwrite(os.path.join(args.output_dir_interpolated, mfile.split('/')[-1]), out_img_interp, [cv.IMWRITE_PNG_COMPRESSION, args.compression])  
     
            if args.log_images:
                self.experiment.log_image(np.asarray(output_img), name=f"{id:05}_nih_filtered_{os.path.basename(mfile)}", image_channels='first')
                self.experiment.log_image(np.asarray(out_img_interp), name=f"{id:05}_nih_smoothed_{os.path.basename(mfile)}", image_channels='first')
            
            self.processed_images +=1

            print(f"\rContours counter - 1: {self.one_contour}, 2: {self.two_contours}, 2+: {self.more_contours}. Rejected - disprop.: {self.high_disproportion}, solidity: {self.rejected_solidity_count}, line: {self.rejected_line_count}. Total rejected:{self.high_disproportion + self.rejected_line_count + self.rejected_solidity_count}/{self.images_count}. Processed: {self.processed_images}/{self.images_count}, {mfile.split('/')[-1]}", end="")

        if args.save_bboxes:
            print(f"\n\n >> Mean bbox size across dataset: {sum(self.ratios) / len(self.ratios):.2f}")     

            filename = os.path.join(args.output, "bounding_boxes.csv")
            with open(filename, 'w') as f:

                write = csv.writer(f)
                write.writerow(['file_name', 'x_min', 'y_min',
                                'x_max', 'y_max', 'mask_size'])
                write.writerows(self.bounding_boxes)
            print(f"Created file with bounding boxes: {filename}")


        if args.rejected_to_csv:
            filename = os.path.join(os.path.join(*args.output_dir.split('/')[:-2]),"rejected_masks.csv")
            with open(filename, 'w') as f:

                write = csv.writer(f)
                write.writerow(['path', 'condition'])
                write.writerows(self.rejected_paths)
            print(f"Created file with rejected images paths: {filename}")

filter = SegmentationMasksFilter(mask_files, args.rejection_conditions)

start_time = time.time()
if args.file:
    filter.filterOneFile(args.file_dir)
else:
    filter.runFiltration()

print("")
print(f"Filtration took: {time.time()-start_time:.0f}s.")

if args.copy_rejected:

    print("")
    print("^^-----")
    print(f"Copying to folders {len(filter.rejected_paths)} rejected images...")

    from shutil import copy2
    for path in filter.rejected_paths:
        if path[1] == "disprop":
            copy2(path[0], "data/nih/rejected_disproportion")
        elif path[1] == "line":
            copy2(path[0], "data/nih/rejected_line_fit")
        elif path[1] == "solidity":
            copy2(path[0], "data/nih/rejected_solidity")
