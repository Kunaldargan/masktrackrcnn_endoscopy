import cv2
import copy
import os
import json
import mmcv
import numpy as np

import math 

path_to_train = "./data/annotations/instances_train_sub.json"
path_to_val = "./data/annotations/instances_val_sub.json"
path_to_images= "./data/train/JPEGImages/"
save_path  = "./data/train/JPEGImages/"

file = open(path_to_train,'r')
data = json.load(file)

val_set_percentage = 10
oversampling_threshold = 0.3

label_sum={}
balanced = []
videos = {} 
LabelId =  {} #
fold_class_wise = {}

counter=0
CATEGORIES = data["categories"]

for cat in CATEGORIES:
	LabelId[cat['id']] = cat['name']

 

#rotate_augs= [10,-10,20,-20,30,-30,40,-40,50,-50,60,-60,70,-70,80,-80,90,-90]

#experiment_1
rotate=False;

#video :dict_keys(['id', 'file_names', 'length', 'width', 'height'])

for vid in data["videos"]:
	video_name = vid['file_names'][0].split("/")[0]

	videos[vid['id']] = video_name
print("training videos",videos)

import random
random.seed(4)
#dict_keys(['info', 'licenses', 'categories', 'videos', 'annotations'])
random.seed(4) 
def create_train_val(data):
	val_set = copy.deepcopy(data)
	train_set = copy.deepcopy(data)

	for vid in data["videos"]:
		print(len(vid["file_names"]))
	k=int(len(data["videos"][0]["file_names"])*val_set_percentage/100);
	
	indices=random.sample(range(len(data["videos"][0]["file_names"])),k)

	sorted(indices)
	train_indices = list(set(range(len(data["videos"][0]["file_names"]))).difference(set(indices)))

	for video_idx,vid in enumerate(train_set["videos"]):
		fn=[]
		fn_val=[]
		for idx,file_name in enumerate(vid["file_names"]):
			if idx in train_indices:
				fn.append(file_name)
			else:
				fn_val.append(file_name)

		train_set["videos"][video_idx]["file_names"] = fn
		train_set["videos"][video_idx]["length"]=len(vid["file_names"])

		val_set["videos"][video_idx]["file_names"]=fn_val
		val_set["videos"][video_idx]["length"]=len(fn_val)
						
	
	for ann_idx, ann in enumerate(train_set["annotations"]):
		segms=[]
		bboxes=[]
		areas=[]
		segms_val=[]
		bboxes_val=[]
		areas_val=[]
		for idx, segm in enumerate(ann["segmentations"]):
			if idx in train_indices:
				segms.append(segm)
				bboxes.append(ann["bboxes"][idx])
				areas.append(ann["areas"][idx])
			else:
				segms_val.append(segm)
				bboxes_val.append(ann["bboxes"][idx])
				areas_val.append(ann["areas"][idx])
		
		train_set["annotations"][ann_idx]["segmentations"] = segms
		train_set["annotations"][ann_idx]["bboxes"] = bboxes
		train_set["annotations"][ann_idx]["areas"] = areas
		val_set["annotations"][ann_idx]["segmentations"]=segms_val
		val_set["annotations"][ann_idx]["bboxes"]=bboxes_val
		val_set["annotations"][ann_idx]["areas"]=areas_val

	
	return train_set,val_set

train_set, val_set=create_train_val(data)
data = copy.deepcopy(train_set)

for key,val in LabelId.items():
        fold_class_wise[val] = 0


for ann in data["annotations"]:#dict_keys(['id', 'video_id', 'category_id', 'segmentations', 'bboxes', 'areas', 'iscrowd'])

        label = LabelId[ann['category_id']];
        for segms in ann['segmentations']:
                if segms:
                        fold_class_wise[label]+=1

max_class = ""
max_class_num = 0

for label, num in fold_class_wise.items():
        if num > max_class_num:
                max_class = label
                max_class_num = num

print(fold_class_wise) #fold wise class items remaining

print("max label: ",max_class,"| occurences: ", max_class_num)


total_images= 0
for vid in data["videos"]:
	total_images+=len(vid["file_names"])


repeat_factors={}
for label, num in fold_class_wise.items():
        reps = fold_class_wise[label]/total_images
        print(label,fold_class_wise[label]/total_images)
        if reps > oversampling_threshold:
                repeat_factors[label] = math.ceil(fold_class_wise[label]/total_images)
        else:
             	repeat_factors[label]=0
print(repeat_factors)
		
for vid in data["videos"]:
	id = vid["id"]
	anns = [] #list of annotations corresponding to video id
	indices=set()
	max_rep = 0;
	for ann in data["annotations"]:
		if ann["video_id"] == id:
			label = LabelId[ann['category_id']]
			anns.append(ann)
			if max_rep < repeat_factors[label]:
				max_rep = repeat_factors[label] 
			if repeat_factors[label]:
				for i,segm in enumerate(ann["segmentations"]):
					if segm:
						indices.add(i)		

	ann_updates = {}
	
	for ann in anns:
		label =  LabelId[ann['category_id']]
		ann_updates[ann["id"]]={"label":label,"segmentations":[],"bboxes":[],"areas":[]}
	
	file_names=[]
	for i,file_name in enumerate(vid["file_names"]):
		if file_name not in file_names:
			file_names.append(file_name)
		for ann in anns:
			label =  LabelId[ann['category_id']]
			ann_id = ann["id"]
			
			ann_updates[ann_id]["segmentations"].append(ann["segmentations"][i])
			ann_updates[ann_id]["bboxes"].append(ann["bboxes"][i])
			ann_updates[ann_id]["areas"].append(ann["areas"][i])
			
			if i in indices:
				for j in range(max_rep):
					segm = ann["segmentations"][i]
					ann_updates[ann_id]["segmentations"].append(ann["segmentations"][i])
					ann_updates[ann_id]["bboxes"].append(ann["bboxes"][i])
					ann_updates[ann_id]["areas"].append(ann["areas"][i])
					
					if segm :
						fold_class_wise[label]+=1

					f = file_name.split(".")
					fname = f[0]+"_aug_"+str(j)+"."+f[-1]
					
					if not os.path.exists(os.path.join(path_to_images,fname)):
						print("path:",os.path.join(path_to_images,file_name))
						img =cv2.imread(os.path.join(path_to_images,file_name))
						cv2.imwrite(os.path.join(path_to_images,fname),img)
						file_names.append(fname)
			
					
				
	for ann in anns:	
		label =  LabelId[ann['category_id']]
		ann_id = ann["id"]
		ann["segmentations"]=ann_updates[ann_id]["segmentations"]
		ann["bboxes"]=ann_updates[ann_id]["bboxes"]
		ann["areas"]=ann_updates[ann_id]["areas"]
	#print(file_names)
	vid["file_names"]=file_names
	vid["length"] = len(file_names)
	#print(fold_class_wise)

test_fold_class_wise = {}

total_images= 0
for vid in train_set["videos"]: 
         total_images+=len(vid["file_names"])

for key,val in LabelId.items():
	test_fold_class_wise[val] = 0

for ann in data["annotations"]:#dict_keys(['id', 'video_id', 'category_id', 'segmentations', 'bboxes', 'areas', 'iscrowd'])
	label = LabelId[ann['category_id']];
	
	for segms in ann['segmentations']:
		if segms:
			test_fold_class_wise[label]+=1


 
print(test_fold_class_wise)
for label, num in test_fold_class_wise.items(): 
	print(label,test_fold_class_wise[label]/total_images)
f = open(path_to_train,'w')
json.dump(data,f) 

val = open(path_to_val,'w')
json.dump(val_set,val)
