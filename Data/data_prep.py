from typing import List, Any

import cv2
import glob
import os
import json
import skimage
import numpy as np
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.filters import try_all_threshold, threshold_triangle, threshold_otsu
from skimage.measure import label, regionprops, find_contours
# from skimage.draw import polygon, polygon_perimeter
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from PIL import Image


base_folder ='/home/dnahak/Thesis/dataset/few-shot/10_shot/Novel/val/images_1/' #'path/to/folder'
image_path = sorted(glob.glob(base_folder+"**/*.jpg"))



class_names = {
    "Apple___Apple_scab": 33,
    'Apple___Black_rot': 34, 'Apple___Cedar_apple_rust': 35, 'Apple___healthy': 36, 'Blueberry___healthy': 37,

    "Cherry_(including_sour)___Powdery_mildew": 0,
    "Cherry_(including_sour)___healthy": 1,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 2,
    "Corn_(maize)___Common_rust_": 3,
    "Corn_(maize)___Northern_Leaf_Blight": 4,
    "Corn_(maize)___healthy": 5,
    "Grape___Black_rot": 6,
    "Grape___Esca_(Black_Measles)": 7,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 8,
    "Grape___healthy": 9,
    "Orange___Haunglongbing_(Citrus_greening)": 10, 'Peach___Bacterial_spot': 11 , 'Peach___healthy':12,

    "Pepper,_bell___Bacterial_spot": 13,
    "Pepper,_bell___healthy": 14,
    "Potato___Early_blight": 15,
    "Potato___Late_blight": 16,
    "Potato___healthy": 17,
    "Raspberry___healthy": 18,
    "Soybean___healthy": 19,
    "Squash___Powdery_mildew": 20,
    "Strawberry___Leaf_scorch": 21,
    "Strawberry___healthy": 22,
    "Tomato___Bacterial_spot": 23,
    "Tomato___Early_blight": 24,
    "Tomato___Late_blight": 25,
    "Tomato___Leaf_Mold": 26,
    "Tomato___Septoria_leaf_spot": 27,
    "Tomato___Spider_mites Two-spotted_spider_mite": 28,
    "Tomato___Target_Spot": 29,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 30,
    "Tomato___Tomato_mosaic_virus": 31,
    "Tomato___healthy": 32
}
# for items in result_dir:
#     path = os.path.join(cat_folder, items)
#     os.mkdir(path)



def mask_for_image(image):

    mask = None
    try:
        thres = threshold_triangle(image)
        thresh_image = image > thres
        mask = np.zeros((image.shape),dtype=np.uint8)
        mask[thresh_image] = 255
    except Exception as error:
        print(f"could not do it{image}")

    return mask

def mask_to_bbox(image):
    mask = mask_for_image(image)
    if mask is not None:


        try:

            regions = regionprops(mask)
            bboxes = []

            for reg in regions:
                Xmin = reg.bbox[1]
                Xmax = reg.bbox[3]
                Ymin = reg.bbox[0]
                Ymax = reg.bbox[2]

                bboxes.append([Ymin, Xmin, Ymax, Xmax])

                return bboxes

        except Exception as error:
            print(f'cant change {image}')
    else:
        return None


def bbox_details(image, class_num):

    mask = mask_for_image(image)

    #try:
    regions = regionprops(mask)



    for reg in regions:
        Xmin = reg.bbox[1]
        Xmax = reg.bbox[3]
        Ymin = reg.bbox[0]
        Ymax = reg.bbox[2]

        w = Xmax - Xmin #height of the bounding box
        h = Ymax - Ymin #width
        x = (Xmax + Xmin) / 2   #(x and y are the obejct co-ordinates)
        y = (Ymax + Ymin)  / 2

        img_width = 256
        img_height = 256

        # need to normalize the values between 0 to 1 for the yolo dataset format
        width = w/img_width
        height = h/img_height
        xc = x/img_width
        yc = y/img_height

        bbox_detail = [xc, yc, width, height]#centroid, width and height


        return bbox_detail

   



for idx, image in enumerate(image_path):
     print(image)

     # try:
     img = imread(image)
     name = image.split("/")[-1].split(".")[0]
     class_name = image.split("/")[-2]
     class_num = class_names[class_name]
     gray_scale = rgb2gray(img)
     bboxes = mask_to_bbox(gray_scale)
     details = bbox_details(gray_scale, class_num)
     

     labels = f"/home/dnahak/Thesis/dataset/few-shot/10_shot/Novel/val/labels/{name}.txt"

     file = open(labels, 'x')
     file.write(f"{class_num} {details[0]} {details[1]} {details[2]} {details[3]}")


