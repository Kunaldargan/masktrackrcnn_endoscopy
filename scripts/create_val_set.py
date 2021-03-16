import cv2
import os
import json
import mmcv
path_to_train = "../data/annotations/instances_train_sub.json"
path_to_images= "../data/endoVis/JPEGImages/"
path_to_save  = "../data/train/JPEGImages/"

file = open(path_to_train,'r')
data = json.load(file)

fold = "0"
val_set_percentage = 20

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

LabelId =  { 1:"Bipolar Forceps",
	     2:"Prograsp Forceps",
	     3:"Large Needle Driver",
	     4:"Vessel Sealer",
             5:"Grasping Retractor",
             6:"Monopolar Curved Scissors",
             7:"Ultrasound Probe"} 

fold_class_wise = {}
 
for key,val in LabelId.items():
	fold_class_wise[val] = 0
		
#print(data["videos"][0].keys())
#print(len(data["annotations"]))

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


print(fold_class_wise)
print("max label: ",max_class,"| occurences: ", max_class_num)
#rotate_augs= [10,-10,20,-20,30,-30,40,-40,50,-50,60,-60,70,-70,80,-80,90,-90]

#experiment_1
rotate=False;

#video :dict_keys(['id', 'file_names', 'length', 'width', 'height'])

def get_video_pos(video_id):
	for pos, vid in enumerate(data["videos"]):
		if vid["id"]==video_id:
			return pos
	print("not found")	

print(data.keys())




def balance(data,fold_class_wise):
	for ann in data["annotations"]:
		pos = get_video_pos(ann['video_id'])
		label = LabelId[ann['category_id']]
		video = data['videos'][pos]
		ann_seg_aug= []
		video_aug_frames = []
		
		if label != max_class and fold_class_wise[label] < max_class_num:
			for i, segms in enumerate(ann['segmentations']):
				ann_seg_aug.append(segms);
				video_aug_frames.append(data['videos'][pos]['file_names'][i])
				iteration = max_class_num-fold_class_wise[label]	
				if segms and iteration:
					fold_class_wise[label]+=1
					if not rotate:
						ann_seg_aug.append(segms)
						fname = data['videos'][pos]['file_names'][i]
						img =cv2.imread( os.path.join(path_to_save,fname))
						print(fname)
						fn = fname.split(".")
						save_fname = fn[0]+"_aug."+fn[-1]
						video_aug_frames.append(save_fname) 
						cv2.imwrite(os.path.join(path_to_save,save_fname),img)
						print(save_fname)
		
			data['videos'][pos]['file_names'] = video_aug_frames
			ann['segmentations']=ann_seg_aug		
	return data, fold_class_wise
	

data, fold_class_wise=balance(data,fold_class_wise)

fold_class_wise= {}
 
for key,val in LabelId.items():
        fold_class_wise[val] = 0



for ann in data["annotations"]:#dict_keys(['id', 'video_id', 'category_id', 'segmentations', 'bboxes', 'areas', 'iscrowd'])
	label = LabelId[ann['category_id']];
	
	for segms in ann['segmentations']:
		if segms:
			fold_class_wise[label]+=1



print(fold_class_wise)














































































































































































































































				
