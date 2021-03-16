from pycocotools import mask as maskUtils
import json
import cv2
import os
import numpy as np


ff = open("output.pkl.json")
data = json.load(ff)
print(len(data))
#print(data[0].keys())
valid_dir = "data/valid/JPEGImages/E1/"
train_dir = "data/train/JPEGImages"
train_annotations= "data/annotations/instances_train_sub.json"
save_dir = "data/E1/"
os.mkdir(save_dir)


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


i = 0

gt_file = open(train_annotations,'r')
data_gt = json.load(gt_file)

for path in sorted(os.listdir(valid_dir)):
	im = cv2.imread(os.path.join(valid_dir,path))
	for key in data:
		if key["video_id"]==1:
			label =  key['category_id']
			segm_pancreas = key['segmentations'][i]
			#segm_cancer = data[3]['segmentations'][i]
	
			#red is cancer predicted
			#green is pancreas predicted
			if label==1:
				color = (0.0,1,0.0)
			if label==2:
                                color =	(0.0,0.0,1)
			if label==3:
                                color =	(1,0.0,0.0)
			if segm_pancreas: 
				mask = maskUtils.decode(segm_pancreas)  # decode directly for prediction : input is rle compressed format < coming from output.pkl>
  
				padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask
				im = apply_mask(im, mask, color).astype(np.uint8)
		"""
		if segm_cancer:
			mask = maskUtils.decode(segm_cancer)
			padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
			padded_mask[1:-1, 1:-1] = mask
			im = apply_mask(im, mask, (0.0,0.0,1)).astype(np.uint8)
		"""	
	cv2.putText(im, "Prediction", (10,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 1)
	print(os.path.join(save_dir,str(i)+".jpg"))        
	cv2.imwrite(os.path.join(save_dir,str(i)+".jpg"),im)
	i = i+1



##############################################################################################################
#			GROUND TRUTH GENERATION								     #
##############################################################################################################

def get_anns(vid):
	for anns in data_gt["annotations"]:
		if anns["video_id"] == vid and anns["category_id"]== 1:
			pancreas = anns["segmentations"]
		if anns["video_id"] == vid and anns["category_id"]== 2:
			cancer = anns["segmentations"]

	return pancreas, cancer

frames_map = {}

def get_frames():
	for v in data_gt["videos"]:
		frames_map[v["id"]] = { "len" :v["length"] , "frames" :v["file_names"] }

get_frames()		


for vid in sorted(os.listdir(train_dir)):
	v_id = vid
	if v_id in frames_map:
		frames_list = frames_map[v_id]["frames"] 
		pancreas, cancer = get_anns(v_id)
		os.mkdir(os.path.join("data/gt",vid)) #Mask ground truth directory by names

		for frame_id in range(len(frames_list)):	
			im_gt = cv2.imread(os.path.join(train_dir,frames_list[frame_id]))
			ht, wt = im_gt.shape[:2] #height , width
				
		
			segm = pancreas[i]
			if segm:
				rles = maskUtils.frPyObjects(segm, ht, wt)
				"""
				What we have is list of polygons: [ polygon ]
				[[x1,...xn] , [y1,...yn]]
				rles: list of rle
				"""
				rle = maskUtils.merge(rles) # combined rle format for the image 			
				mask = maskUtils.decode(rle) # decode the rle
				padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask
				im_gt = apply_mask(im_gt, mask, (0.0,1,0.0)).astype(np.uint8) # Test Padded Mask 

			segm = cancer[i]
			if segm:
				rles = maskUtils.frPyObjects(segm, ht, wt)
				rle = maskUtils.merge(rles)
				mask = maskUtils.decode(rle)
				padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask
				im_gt = apply_mask(im_gt, mask, (0.0,0.0,1)).astype(np.uint8)

			cv2.putText(im_gt, "Ground Truth", (10,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 1)

			file_name = frames_list[frame_id].split("/")[-1]
			cv2.imwrite(os.path.join("data/gt/"+vid,file_name),im_gt)
