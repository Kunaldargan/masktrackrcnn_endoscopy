import os
import pprint
import json

file_train = open("../annotations_dir/endoVis_fold_3/instances_train_sub.json",'r')
file_val = open("../annotations_dir/endoVis_fold_3/instances_val_sub.json",'r')


data_train = json.load(file_train)
data_val = json.load(file_val)

train_map = {}
print(data_train.keys())

with open("../labels.txt",'w') as f:
	for cat in data_train['categories']:
		f.write(cat["name"]+"\n")	
	
for vid in data_train["videos"]:
	id = vid["id"]
	filename = vid["file_names"][0].split("/")[0]
	train_map[id] = filename

fp = open("../train_map.json",'w')
json.dump(train_map,fp)
val_map = {}
print(data_val.keys())
for vid in data_val["videos"]:
        id = vid["id"]
        filename = vid["file_names"][0].split("/")[0]
        val_map[id] =	filename
fp = open("../val_map.json",'w')
json.dump(val_map,fp)

