from pycocotools import mask as maskUtils
import json
import cv2
import os
import numpy as np


ff = open("output.pkl.json")
data = json.load(ff)
print(len(data))
#print(data[0].keys())
valid_dir = "data/valid/JPEGImages/421/"
valid_annotations= "data/annotations/instances_val_sub.json"
save_dir = "data/save/"

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

gt_file = open(valid_annotations,'r')
data_gt = json.load(gt_file)

for path in sorted(os.listdir(valid_dir)):
	im = cv2.imread(os.path.join(valid_dir,path))
	segm_pancreas = data[2]['segmentations'][i]
	segm_cancer = data[3]['segmentations'][i]
	
	#red is cancer predicted
	#green is pancreas predicted

	if segm_pancreas: 
		mask = maskUtils.decode(segm_pancreas)
		padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		im = apply_mask(im, mask, (0.0,1,0.0)).astype(np.uint8)
	
	if segm_cancer:
		mask = maskUtils.decode(segm_cancer)
		padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		im = apply_mask(im, mask, (0.0,0.0,1)).astype(np.uint8)

	cv2.putText(im, "Prediction", (10,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 1)
            
	cv2.imwrite(os.path.join(save_dir,str(i)+".jpg"),im)
	i = i+1


"""
   	print(data[0]["score"])
        print(data[0]["category_id"])
        print(data[1]["score"])
        print(data[1]["category_id"])
        print(data[2]["score"])
        print(data[2]["category_id"])
        print(data[3]["score"])
        print(data[3]["category_id"])
        print(data[4]["score"])
        print(data[4]["category_id"])
        print(data[5]["score"])
        print(data[5]["category_id"])


0.18806762993335724
1
0.30180230736732483
1
0.7768524289131165
1
0.22801423072814941
2
0.1430695354938507
2
"""

i = 0
pancreas = data_gt["annotations"][0]
cancer = data_gt["annotations"][1]

for path in sorted(os.listdir(valid_dir)):
	im_gt = cv2.imread(os.path.join(valid_dir,path))
	ht, wt = im_gt.shape[:2]
	
	segm = pancreas['segmentations'][i]
	if segm:
		rles = maskUtils.frPyObjects(segm, ht, wt)
		rle = maskUtils.merge(rles)			
		mask = maskUtils.decode(rle)
		padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		im_gt = apply_mask(im_gt, mask, (0.0,1,0.0)).astype(np.uint8)

	segm = cancer['segmentations'][i]
	if segm:
		rles = maskUtils.frPyObjects(segm, ht, wt)
		rle = maskUtils.merge(rles)
		mask = maskUtils.decode(rle)
		padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		im_gt = apply_mask(im_gt, mask, (0.0,0.0,1)).astype(np.uint8)

	cv2.putText(im_gt, "Ground Truth", (10,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 1)


	cv2.imwrite(os.path.join("data/gt",str(i)+".jpg"),im_gt)
	i = i+1
