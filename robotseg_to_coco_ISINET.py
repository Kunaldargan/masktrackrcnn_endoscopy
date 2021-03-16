# !/usr/bin/env python3
# Modified from https://github.com/waspinator/pycococreator/

import cv2
import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
import collections
from pycococreatortools import pycococreatortools


convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    #image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

def create_annotation_info(binary_mask, image_size=None, tolerance=2):

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = int(mask.area(binary_mask_encoded))
    bounding_box = mask.toBbox(binary_mask_encoded).astype(np.int32).tolist()
    segmentation = binary_mask_to_rle(binary_mask)
    #segmentation = binary_mask_to_polygon(binary_mask, tolerance)
    
    if area < 1:
       area = None
       bounding_box = None
       segmentation = None

    return area, bounding_box, segmentation
       

import argparse
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
#from pycococreatortools import pycococreatortools
from tqdm import tqdm



def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description="Convert robotic segmentation dataset into COCO format"
    )

    parser.add_argument(
        "--root-dir",
        dest="root_dir",
        required=True,
        help="Complete path to image root",
    )
    
    parser.add_argument(
        "--dataset",
        dest="dataset",
        required=True,
        help="Dataset name",
    )

    parser.add_argument(
        "--fold",
        dest="fold_num",
        required=False,
        type = int,
        help="fold number for Train and Val",
    )

    parser.add_argument(
        "--visualize",
        dest="vis",
        required=False,
        help="Debug save masks [False default]",
    )


   
    return parser.parse_args()


args = parse_args()
print("Called with args:")
print(args)

# setup paths to data and annotations
IMAGE_DIR = args.root_dir
DATASET = args.dataset
FOLD_NUM = args.fold_num


INFO = {
    "description": "Robotic Instrument Type Segmentation",
    "url": "",
    "year": DATASET,
    "contributor": "C.Gonzalez, L. Bravo-Sanchez, P. Arbelaez",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [{"id": 1, "name": "", "url": ""}]

FOLDS = {0: [1, 3], 
	 1: [2, 5], 
	 2: [4, 8], 
	 3: [6, 7]}

CATEGORIES = [
    {"id": 1, "name": "Bipolar Forceps", "supercategory": "Instrument"},
    {"id": 2, "name": "Prograsp Forceps", "supercategory": "Instrument"},
    {"id": 3, "name": "Large Needle Driver", "supercategory": "Instrument"},
    {"id": 4, "name": "Vessel Sealer", "supercategory": "Instrument"},
    {"id": 5, "name": "Grasping Retractor", "supercategory": "Instrument"},
    {"id": 6, "name": "Monopolar Curved Scissors", "supercategory": "Instrument"},
    {"id": 7, "name": "Ultrasound Probe", "supercategory": "Instrument"},
    {"id": 8, "name": "Suction Instrument", "supercategory": "Instrument"},
    {"id": 9, "name": "Clip Applier", "supercategory": "Instrument"},
]

if DATASET == "2017":
    CATEGORIES = CATEGORIES[:7]
elif DATASET == "2018":
    CATEGORIES = CATEGORIES[:3] + CATEGORIES[5:]
    for i, c in enumerate(CATEGORIES):
        c['id'] = i + 1

def filter_for_jpeg(root, files):
    file_types = ["*.jpeg", "*.jpg"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_png(root, files):
    file_types = ["*.jpg"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, image_filename):
    #file_types = ["*.png"]
    #file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[
        0
    ]
    #print(basename_no_extension)
    file_name_prefix = basename_no_extension + ".png"

    #files = [os.path.join(root, f) for f in files]
    #files = [f for f in files if re.match(file_types, f)]

    files_match=[]

    for dir in os.listdir(root):
    	if os.path.exists(os.path.join(root,dir+"/"+file_name_prefix)):
    		files_match.append(os.path.join(root,dir+"/"+file_name_prefix))
	  
    
    return files_match

def get_label(label):
    label = label.replace("_"," ")
    for cat in CATEGORIES:
        if cat["name"] in label:
            return label, cat["id"];

    return None,None #To be handled

	
	

def get_images(IMAGE_DIR, image_id):
    
    images=[]

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_png(root, files)
        image_files.sort()  # ensure order

        # go through each image
        for image_filename in tqdm(image_files):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size
            )
            images.append(image_info)
            image_id +=1
    return images,image_id


def data_split(FOLD_NUM):
    val = FOLDS[FOLD_NUM];
    train = list(set(range(1,9)).difference(set(val)))
    
    val = ["instrument_dataset_"+str(x) for x in val]
    train = ["instrument_dataset_"+str(x) for x in train]
    
    return train, val
    
"""
annotation{
    "id" : int, 
    "video_id" : int, 
    "category_id" : int, 
    "segmentations" : [RLE or [polygon] or None], 
    "areas" : [float or None], 
    "bboxes" : [[x,y,width,height] or None], 
    "iscrowd" : 0 or 1,
}
"""
def main():

	image_id = 1
	segmentation_id = 1
	VIDEOS =[]
	ANNOTATIONS =[]
	IMAGES_VAL =[]
	IMAGES_TRAIN =[]
	ANNOTATIONS_VAL =[]
	ANNOTATIONS_TRAIN =[]

	root_dir = IMAGE_DIR
	train, val = data_split(FOLD_NUM)

	# filter for jpeg images
	for dir in os.listdir(root_dir):
		print(dir)
		path = os.path.join(root_dir,dir)+"/left_frames/"
		ANNOTATION_DIR = os.path.join(root_dir,dir)+"/ground_truth/"
		#print(path)
		# filter for jpeg images
		for root, _, files in os.walk(path):
			
			image_files = [os.path.join(path,f) for f in files]#filter_for_png(root, files)
			
			image_files.sort()  # ensure order
			#print(image_files)
			# go through each image
			for image_filename in image_files: #tqdm(image_files):
				image = Image.open(image_filename)
				#print(image_filename)
				image_info = pycococreatortools.create_image_info(
					image_id, os.path.basename(image_filename), image.size
				)
				image_info["file_name"]=os.path.join(dir,image_info["file_name"])			

				# filter for associated png annotations
				
				annotation_files = filter_for_annotations(
				ANNOTATION_DIR,image_filename
				)
				#print("here",annotation_files)
				if len(annotation_files):
					if dir not in train:
						IMAGES_VAL.append(image_info)
					else:
						IMAGES_TRAIN.append(image_info)
					image_id +=1
					# go through each associated annotation
					for annotation_filename in annotation_files:
						
						label,class_id = get_label(annotation_filename.split("/")[-2])
						if not class_id:
							print(annotation_filename)
							print(label,class_id)

						category_info = {
							"id": class_id							
							}
						binary_mask = cv2.imread(annotation_filename,0).astype(np.uint8)
						area,bbox,segmentation = create_annotation_info(binary_mask,None)
						if area:
							annotation_info=  {
			        				"id": segmentation_id,
        							"image_id": image_id,
 		       						"category_id": class_id,
        							"iscrowd":None,
        							"area": area,
        							"bbox": bbox,
							        "segmentation": segmentation,
        							"width": binary_mask.shape[1],
        							"height": binary_mask.shape[0],
    								} 
							segmentation_id+=1
							if dir not in train:          
								ANNOTATIONS_VAL.append(annotation_info)
							else:
								ANNOTATIONS_TRAIN.append(annotation_info)

	coco_output_train = {
		"info": INFO,
		"licenses": LICENSES,
		"categories": CATEGORIES,
		"images": IMAGES_TRAIN,
		"annotations": ANNOTATIONS_TRAIN,
	}       

	coco_output = {
		"info": INFO,
		"licenses": LICENSES,
		"categories": CATEGORIES,
		"images": IMAGES_VAL,
		"annotations": ANNOTATIONS_VAL,
	}

	with open( "instances_train_sub.json", "w") as output_json_file:
		json.dump(coco_output_train, output_json_file, indent=4)

	with open( "instances_val_sub.json", "w") as output_json_file:
		json.dump(coco_output, output_json_file, indent=4)


if __name__ == "__main__":
    main()
  
