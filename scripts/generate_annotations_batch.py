import os
import numpy as np

import json

from PIL import Image
import cv2

def read_labels(path):
	labels = []
	i=0
	file = open(path,'r')
	for line in file.readlines():
		if "__ignore__" not in line and "_background_" not in line:
			category = {
			            "id" : i,
            			    "name" : line.strip(),
            			    "supercategory" : "object",
        			   }
			i = i+1;
			labels.append(category)
	return labels
	
vids=[""]

#vids = ["T10_2.avi","T1_1.avi","T8_1.avi","T11_3.avi","T3_2.avi","T9_1.avi"]

# set home directory and data directory 
HOME = "./Endoscopy"
save_path = "instances_val_sub_"+ "shortlisted_biopsy_videos_15" +".json"

#get info
info = {'description': 'YouTube-VOS',
 'url': 'https://youtube-vos.org/home',
 'version': '1.0',
 'year': None,
 'contributor': None,
 'date_created': None}

#get categories
categories = []

# define category for pancreas
category = {
            "id" : 1, 
            "name" : "disk", 
            "supercategory" : "object",
        }

categories.append(category)

# define category for cancer
category = {
            "id" : 2, 
            "name" : "tool", 
            "supercategory" : "object",
        }

categories.append(category)

category = {
            "id" : 3, 
            "name" : "endoscope", 
            "supercategory" : "object",
        }

categories.append(category)

categories=read_labels("labels.txt") #label file path // close it if not required

print(categories)

#get video annotations
import collections
import glob


from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import cv2

def segment(points):
        # Make a polygon and simplify it
        contour = np.array(points)
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()      
        x, y, max_x, max_y = poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = [x, y, width, height]
        area =  poly.area

        segmentation = [int(x) for x in segmentation]
        bbox = [int(x) for x in bbox]
        segm = {
            'segmentation': [segmentation],
            'bbox': bbox,
            'area': int(area)
        }
        return segm
        
def find_category(label):
    if 'disk'in label:
        return 1
    elif 'tool' in label:
        return 2
    else:
        return 3
        
def get_annotations(label_file="/home/kunaldargan/ROBO_SURGERY/MaskRCNNData/data_maskrcnn_clips/E1/E1_t1/", 
                    video_id=1, annotation_id=0):
    '''return a dict of annotations from the video.
    three annotations one for disk, tool and endoscope'''
    
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
    sub_annotation = {}
    meta = {}
    frame_numbers = set()
    labels = set()
    
    for file in sorted(os.listdir(label_file)):
        if file.endswith('.json'): 
            
            print(os.path.join(label_file,file))
            fp = open(os.path.join(label_file,file),'r')
            data = json.load(fp)
            val = file.split(".json")[0]
            for shapes in data["shapes"]:                     
                points = shapes["points"]
                height = data["imageHeight"]
                width = data["imageWidth"]
                segm = segment(points)
                area = segm['area']
                bbox = segm['bbox']
                segmentation = segm['segmentation']
                if shapes["label"] not in meta:
                    meta[shapes["label"]] = {}
                    meta[shapes["label"]][val]= (height, width,area,bbox,segmentation)
                else:
                    meta[shapes["label"]][val]= (height, width,area,bbox,segmentation)
                frame_numbers.add(val)
                labels.add(shapes["label"])
    
    for lbl in labels:
        annotation_id += 1 
        sub_annotation[lbl]={
            "id" : annotation_id, 
            "video_id" : video_id, 
            "category_id" : find_category(lbl), 
            "segmentations" : [], 
            "areas" : [], 
            "bboxes" : [], 
            "iscrowd" : 0
        }
        
    """ Structure
    meta --> keys=lbls :{ framenumber : (tuple )}  
    meta["disk1"]={"0001": (height, width,area,bbox,segmentation)
                    "0002":(height, width,area,bbox,segmentation)
    }
    """
    for key, value in meta.items():
        for fn in sorted(frame_numbers):
            if fn not in value.keys():
                sub_annotation[key]['segmentations'].append(None)
                sub_annotation[key]['areas'].append(None)
                sub_annotation[key]['bboxes'].append(None)
            else:
                sub_annotation[key]['areas'].append(value[fn][2])
                sub_annotation[key]['bboxes'].append(value[fn][3])
                sub_annotation[key]['segmentations'].append(value[fn][4])
       
                
    return sub_annotation, annotation_id 


def get_video(filenames, video_id, valid_im):
    video = collections.defaultdict()
    video["id"] = video_id
    video["file_names"] = filenames
    video["length"] = len(video["file_names"])
    h, w = valid_im.shape[:2]
    video["width"] = w
    video["height"] = h
    
    return video
            
from PIL import Image

VAL_MAP = []
VAL_VIDEOS = []
VAL_ANNOTATIONS = []
vid_id=0
ann_id=0


for TEST_VIDEO in os.listdir(HOME):
	cap = cv2.VideoCapture(os.path.join(HOME,TEST_VIDEO))
	print("Video : ",os.path.join(HOME,TEST_VIDEO))
	
	VIDEO_NAME=TEST_VIDEO.split(".")[0]
	ann_id += 1
	val_counter = 0

	filenames = []
	vid_id += 1
	dummy = {
            "id" : ann_id,
            "video_id" : vid_id,
            "category_id" : 1,
            "segmentations" : [],
            "areas" : [],
            "bboxes" : [],
            "iscrowd" : 0,
	}

	while(cap.isOpened()):
		ret, frame = cap.read()
		val_counter += 1
		if ret :
			if not (val_counter+2)%3:
				filenames.append(f"{VIDEO_NAME}/{val_counter}"+".jpg")
				dummy["segmentations"].append([[487, 302, 503, 0, 470, 0, 456, 222, 455, 303, 487, 302]])
				dummy["areas"].append(1)
				dummy["bboxes"].append([455, 0, 48, 303]) #dummy values in coco format
				valid_im = frame 
		else:
			break

	VAL_ANNOTATIONS.append(dummy)
	video = get_video(filenames, vid_id, valid_im)
	VAL_VIDEOS.append(video)

	print(video)
	cap.release()

val_json = {"info": info,
            "videos": VAL_VIDEOS,
            "annotations": VAL_ANNOTATIONS,
            "categories": categories}
with open(save_path, "w") as outfile:  
    json.dump(val_json, outfile) 


            
