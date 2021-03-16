import os
import json


root =  "/scratch/cse/msr/siy207566/EndoVis_2018/"

for dir in sorted(os.listdir(root)):
	for seq_dir in os.listdir(os.path.join(root,dir)):
		if "seq" in seq_dir:
			labels = os.path.join(root+dir+"/"+seq_dir,"labels")
			frames = os.path.join(root+dir+"/"+seq_dir,"left_frames")
			
			
