# !/usr/bin/env python3
# Modified from https://github.com/waspinator/pycococreator/


import os
import re
import cv2
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask
import collections

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

    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    #print(len(binary_mask_encoded))
    area = int(mask.area(binary_mask_encoded))
    bounding_box = mask.toBbox(binary_mask_encoded).astype(np.int32).tolist()
    segmentation = binary_mask_to_rle(binary_mask)
    #segmentation = binary_mask_to_polygon(binary_mask_encoded[0], tolerance)
    
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
        required=False,
        help="Dataset name(Year)",
	default="2017",
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


def filter_for_annotations(root, files, image_filename):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[
        0
    ]
    file_name_prefix = basename_no_extension + ".*"
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [
        f
        for f in files
        if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])
    ]

    return files

def get_label(label):
    label = label.replace("_"," ")
    print(label)
    for cat in CATEGORIES:
        if cat["name"] in label:
            return label, cat["id"];
    
    return None,None #To be handled
	
	
def get_video(folder, video_id, video_file_name=""):

	#print(folder)
	counter = 0
	videos = []

	for i,file in enumerate(sorted(os.listdir(folder))):

		if i%5==0:
			video = collections.defaultdict()
			video["id"] = video_id
			video_id+=1
			video["file_names"] = []
			if video_file_name == "":
				video_name = "default"
			else:
				video_name = video_file_name
			counter+=1
			videos.append(video)

		if file.endswith('.png'):
			file_name = f"{video_name}/{file}"
			videos[counter-1]["file_names"].append(file_name)
	
	#print(videos)	
	for video in videos:
		video["length"] = len(video["file_names"])
		im = Image.open(os.path.join(folder,video["file_names"][0].split("/")[-1]))
		width, height = im.size
		video["width"] = width
		video["height"] = height   
    
	return videos,video_id


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
def verify(video_meta,ANNOTATION_DIR,Ann_Path):
	verified_videos=[]
	for video in video_meta:
		file_names = []
		for file_name in video["file_names"]:
			fn = file_name.split("/")[-1]
			check = False;
			for label in Ann_Path:
				if os.path.exists(os.path.join(ANNOTATION_DIR,label+"/"+fn)):
					annotation_filename = os.path.join(ANNOTATION_DIR,label+"/"+fn)
					binary_mask = cv2.imread(annotation_filename,0).astype(np.uint8)
					#area,bbox,seg=create_annotation_info(binary_mask)
					if not np.all(binary_mask==0):
						check=True
			if check:
				file_names.append(file_name)
				#print(file_name)
			else:
				print("############### null frame ################# :",file_name)

		
		video["file_names"] = file_names
		video["length"] = len(video["file_names"])
		
		if video["length"]>1:
                       verified_videos.append(video)

		if video["length"]==1:
			video["file_names"].append(video["file_names"][0])
			video["length"] = len(video["file_names"])
			verified_videos.append(video)
		
	return verified_videos;

def main():

    val_counter = 1
    segmentation_id = 1
    VIDEOS_VAL =[]
    VIDEOS_TRAIN =[]
    ANNOTATIONS_VAL =[]
    ANNOTATIONS_TRAIN =[]
    root = IMAGE_DIR
    train, val = data_split(FOLD_NUM)
    
    # filter for jpeg images
    for dir in sorted(os.listdir(root)):
        print(dir)
        path = os.path.join(root,dir)+"/left_frames/"
        video_meta,val_counter = get_video(path, val_counter, dir)
        video_meta_verified =[]

        ANNOTATION_DIR = os.path.join(root,dir)+"/ground_truth/"
        Ann_Path = sorted(os.listdir(ANNOTATION_DIR)) #filter for useful labels
        print("labels:",Ann_Path)
        video_meta = verify(video_meta,ANNOTATION_DIR,Ann_Path)

        for video in video_meta:
            for lbl in Ann_Path:
                label, class_id = get_label(lbl)
                ann={
                            "id" : segmentation_id,
                            "video_id" : video["id"],
                            "category_id" : class_id,
                            "segmentations":[],
                            "bboxes":[],
                            "areas":[],
                            "iscrowd" : 0,
                }
                segmentation_id+=1
                for file_name in video["file_names"]:
                    fn = file_name.split("/")[-1]
                    fn_class = file_name.split("/")[0]
                                        
                    if os.path.exists(os.path.join(ANNOTATION_DIR,lbl+"/"+fn)):
                        #print(os.path.join(ANNOTATION_DIR,lbl+"/"+fn))
                        annotation_filename = os.path.join(ANNOTATION_DIR,lbl+"/"+fn)
                        binary_mask = cv2.imread(annotation_filename,0).astype(np.uint8)
                        area, bbox, segm = create_annotation_info(binary_mask,binary_mask.size)
                        ann["areas"].append(area)
                        ann["bboxes"].append(bbox)
                        ann["segmentations"].append(segm)
                           
                if dir not in train:          
                     ANNOTATIONS_VAL.append(ann)
                else:
                     ANNOTATIONS_TRAIN.append(ann)

        if dir not in train:
            VIDEOS_VAL+=video_meta
        else:
            VIDEOS_TRAIN+=video_meta

    coco_output_train = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "videos": VIDEOS_TRAIN,
        "annotations": ANNOTATIONS_TRAIN,
    }       
 
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "videos": VIDEOS_VAL,
        "annotations": ANNOTATIONS_VAL,
    }

    with open( "instances_train_sub.json", "w") as output_json_file:
        json.dump(coco_output_train, output_json_file, indent=4)

    with open( "instances_val_sub.json", "w") as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)


if __name__ == "__main__":
    main()
