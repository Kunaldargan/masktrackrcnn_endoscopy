import os
import json

fn = open("val_map.json",'r')
data = json.load(fn)

for key, val in data.items():
	os.system("python annotate_valid.py "+val+" "+key)
