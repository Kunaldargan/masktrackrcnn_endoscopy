from pycocotools import mask as maskUtils
import json
import cv2
import os
import numpy as np

#vids = ["Dr_Gaurav_Varshney_0deg_LeftTilt_endo.avi","T10_2.avi","T1_1.avi","T8_1.avi",
#"Dr_Gopal_0Deg_RT_Aux.avi,"T11_3.avi","T3_2.avi","T9_1.avi"]

ff = open("output_labels/output_endosurgery.pkl.json")
data = json.load(ff)
print(len(data))
#print(data[0].keys())
valid_dir = "data/valid/JPEGImages/endosurgery/"
train_dir = "data/train/JPEGImages"
train_annotations= "data/annotations/instances_val_sub.json"
save_dir = "data/endosurgery/"
os.mkdir(save_dir)


colors = [(0.0,1,0.0), (0.5,1,0.0), (0.0,1,0.5), (0.5,1,0.5), (0.5,0.5,0.1), (1,0.5,0.5)]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    #mask = mask[0:1024,0:1024]
    #image = image[0:1024, 0:1024]#cv2.resize(image,(1024,1280), interpolation = cv2.INTER_NEAREST)
    
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


i = 0

gt_file = open(train_annotations,'r')
data_gt = json.load(gt_file)
files = os.listdir(valid_dir)
files = [int(x.split(".")[0]) for x in files] 
files.sort()
lsorted = [str(x)+".jpg" for x in files]
print(lsorted)

for path in lsorted:
	path_int = int(path.split(".")[0])
	path = '{:05}.jpg'.format(path_int) 
	im = cv2.imread(os.path.join(valid_dir,path))
	print(os.path.join(valid_dir,path))
	m = np.zeros((1280,1024))
	for idx, key in enumerate(data):
		if key["video_id"]==1:
			label =  key['category_id']
			segm = key['segmentations'][i]

			if label==1:
				color = colors[idx%6]
			if label==2:
                                color =	(0.0,0.0,1)
			if label==3:
                                color =	(1,0.0,0.0)
			if segm: 
				mask = maskUtils.decode(segm)  # decode directly for prediction : input is rle compressed format < coming from output.pkl>  
				#print(mask.shape)
				#print(im.shape)
				padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
				padded_mask[1:-1, 1:-1] = mask
				im = apply_mask(im, mask, color).astype(np.uint8)
	#mask = mask[0:540,0:720]	
	m += 255*cv2.transpose(mask)
	cv2.imwrite(os.path.join(save_dir,str(i)+"_mask.jpg"),m)
	cv2.putText(im, "Prediction : "+path, (10,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 1)
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
