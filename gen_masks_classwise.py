import os
import cv2
import numpy as np
import json

root = "data/train/JPEGImages/"

for dir in os.listdir(root):
	path = os.path.join(root,dir+'/jsons')
	imgs = os.path.join(root,dir+'/images')
	save_dir = os.path.join(root,dir+"/ground_truth")
	os.mkdir(save_dir)
	print(os.path.join(root,dir))

	for img_path in os.listdir(imgs):
		print(img_path)
		img = cv2.imread(imgs+"/"+img_path,1)
		height, width = img.shape[:2]
		lbl = img_path.split(".")[0]+".json"
		file = open(path+"/"+lbl,'r')
		data = json.load(file)
		
		shapes = data["shapes"]
		masks = {}
		for shape in shapes:
			cls = shape["label"]		 
			
			if cls not in masks.keys():
				mask = np.zeros([height, width], dtype=np.uint8)
				pts = shape["points"]
				cv2.drawContours(mask,[np.array(pts).astype(int)], -1, (255,255,255), -1)
				masks[cls] = mask	
			else:	
				mask = points[cls]
				pts = shape["points"]
				cv2.drawContours(mask,np.array(pts).astype(int), -1, (255,255,255), -1)
				masks[cls] = mask

				
		for cls in masks.keys():
			if not os.path.exists(os.path.join(save_dir,cls)):
                                os.mkdir(os.path.join(save_dir,cls))
			cv2.imwrite(os.path.join(save_dir,cls+"/"+img_path),masks[cls])

