import os
import sys

config_file = "masktrack_rcnn_r50_fpn_1x_youtubevos.py"
weight_file_path ="/scratch/cse/msr/siy207566/checkpoints/work_dirs_endoVis_fold0/masktrack_rcnn_r50_fpn_1x_youtubevos"
output_sufix = "endoVis_fold0"
output_labels = "/scratch/cse/msr/siy207566/output_labels/"

for epoch in os.listdir(weight_file_path): 
	weight_file=os.path.join(weight_file_path, epoch)
	epoch = epoch.split(".")[0]+".pkl"
	command = "python3 tools/test_video.py configs/"+config_file+" "+weight_file+" --out "+"output_"+output_sufix+"_"+epoch+" --eval segm --save_path ./data/save/"
	print(command)
	os.system(command)
