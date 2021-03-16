import os
import json

root="/home/cse/msr/siy207566/MaskTrackRCNN/configs/endoVis"
save_dir="/scratch/cse/msr/siy207566/checkpoints/with_blur/work_dirs_with_blur_endoVis_"

for folder in sorted(os.listdir(root)):
	fn = folder[-1]
	save_path = save_dir+folder
	if not os.path.exists(save_path):
		os.mkdir(save_path)
		print(save_path)		
	os.system("cp ./annotations_dir/endoVis_fold_"+fn+"/* ./data/annotations/ ")
	for epochs in sorted(os.listdir(os.path.join(root,folder))):
		path_to_config = os.path.join(root,folder+"/"+epochs)
		print("Command:","CUDA_VISIBLE_DEVICES=0,1 python3 tools/train.py "+path_to_config+" --gpus 2")
		os.system("CUDA_VISIBLE_DEVICES=0,1 python3 tools/train.py "+path_to_config+" --gpus 2")
