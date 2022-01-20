
import os
import cv2
import pandas as pd
import numpy as np
import math
import csv

dataset_df = pd.read_csv(os.path.join('csv_files', 'v7darwin_dataset.csv'))
mask_paths = [os.path.join('v7darwin', 'masks', filename) for filename in dataset_df['mask'].tolist()]
image_paths = [os.path.join('v7darwin', 'images', filename) for filename in dataset_df['file_name'].tolist()]

def get_bounding_boxes_perfect_masks():

    bounding_boxes_list = []
    for idx, file in enumerate(mask_paths):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        rect = cv2.boundingRect(np.concatenate(contours,axis=0))
        x_min = rect[0]
        y_min = rect[1]
        x_max = x_min + rect[2]
        y_max = y_min + rect[3]       
            
        #albumentations format for corners of bounding box
        bb = [x_min, y_min, x_max, y_max]
        divs = [width, height, width, height]    
        bb_normalized = [float(pos) / div for pos, div in zip(bb, divs)]
        
        if bb_normalized[0] < 0.01:
            bb_normalized[0] = 0.01
        if bb_normalized[1] < 0.01:
            bb_normalized[1] = 0.01
        if bb_normalized[2] > 0.99:
            bb_normalized[2] = 0.99
        if bb_normalized[3] > 0.99:
            bb_normalized[3] = 0.99

        out = [file.split('/')[-1], *bb_normalized,width,height]
        bounding_boxes_list.append(out)

        print(f"\r{idx}/{len(mask_paths)}", end="")
    return bounding_boxes_list

def save_boxes():
    bounding_boxes_list = get_bounding_boxes_perfect_masks()

    filename = os.path.join('csv_files', 'darwin_masks_bounding_boxes.csv')
    with open(filename, 'w') as f:

        write = csv.writer(f)
        write.writerow(['file_name', 'x_min', 'y_min',
                        'x_max', 'y_max', 'width', 'height'])
        write.writerows(bounding_boxes_list)
    print(f"Created file with bounding boxes: {filename}")

def scale_cnt(cnt, scale=0.75):
    M = cv2.moments(cnt)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        return os.error

    # cv2.fillPoly(canvas, cnt, 255)
    # cv2.circle(canvas, (cX,cY), 3, 128, thickness=-1)
    
    cnt_norm = cnt - [cX, cY]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cX, cY]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def approximate_contour_debug(c, img_shape):

    image = np.zeros(img_shape, dtype=np.uint8)
    
    for eps in np.linspace(0.001, 0.05, 10):
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps * peri, True)
        # draw the approximated contour on the image
        output = image.copy()
        cv2.drawContours(output, [approx], -1, (255, 255, 255), 3)
        text = f"eps={eps:.4f}, num_pts={len(approx)}"
        cv2.putText(output, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # show the approximated contour image
        print("[INFO] " + text)
        cv2.imshow("Approximated Contour", output)
        cv2.waitKey(0)


def approximate_contour(cnt, eps=0.0025):
    peri = cv2.arcLength(cnt, True)
    approx_cnt = cv2.approxPolyDP(cnt, eps * peri, True)
    return approx_cnt

def save_resized_images_and_masks():
    folder = 'v7darwin_1024'
    # inner and outer scale
    scales = [0.8, 0.9]
    
    for i in range(len(mask_paths)):
        img = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)  
        img = cv2.resize(img, (1024,1024), interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (1024,1024), interpolation = cv2.INTER_AREA)

        # there are some images with segmented electronics like heart-stimulators etc. with gray intensity
        # remove gray class by tresholding
        ret, mask = cv2.threshold(mask,200,255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        scaled_mask = np.zeros(mask.shape, dtype=np.uint8)

        # remove compression artifacts or small discontinuities 
        contours = [cnt for cnt in contours if len(cnt) > 200]

        # two lungs case
        if len(contours)==2:
            for cnt in contours:
                
                cnt = approximate_contour(cnt, eps=0.005)
                cnt_scaled = scale_cnt(cnt, scale=scales[1])
                cv2.drawContours(
                    scaled_mask, [cnt_scaled], -1, color=(255, 255, 255), thickness=cv2.FILLED)
                cnt_scaled = scale_cnt(cnt, scale=scales[0])
                cv2.drawContours(
                    scaled_mask, [cnt_scaled], -1, color=(0, 0, 0), thickness=cv2.FILLED)
        #one lung case
        elif len(contours)==1:
                cnt = approximate_contour(contours[0], eps=0.003)
                cnt_scaled = scale_cnt(cnt, scale=scales[1])
                cv2.drawContours(
                    scaled_mask, [cnt_scaled], -1, color=(255, 255, 255), thickness=cv2.FILLED)
                cnt_scaled = scale_cnt(cnt, scale=scales[0])
                cv2.drawContours(
                    scaled_mask, [cnt_scaled], -1, color=(0, 0, 0), thickness=cv2.FILLED)
        else:
            scaled_mask = mask
            print(f"problem with {mask_paths[i]} contour, copying original mask image, contour size {len(contours)}")

        cv2.imwrite(os.path.join(folder, 'images', image_paths[i].split('/')[-1]), img)
        cv2.imwrite(os.path.join(folder, 'masks', mask_paths[i].split('/')[-1]), mask)
        cv2.imwrite(os.path.join(folder, 'masks_draw', mask_paths[i].split('/')[-1]), scaled_mask)

        #save progress
        print(f"\r{i}/{len(mask_paths)}_{mask_paths[i].split('/')[-1]}", end="")

        ## debug
        # for cnt in contours:
        #     cv2.fillPoly(scaled_mask, cnt, 255) # draws only outer contour
        # cv2.imshow("debug draw area", scaled_mask)
        # cv2.waitKey(0)


if __name__ == "__main__":
    save_resized_images_and_masks()


