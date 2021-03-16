import os
import json

file = open("data/annotations/instances_val_sub.json",'r')
data = json.load(file)

videos = data["videos"]
id_map={}

id = 1
for vid in videos:
	id_map[vid['id']] = id
	vid['id'] = id
	id+=1

for ann in data['annotations']:
	ann['video_id'] = id_map[ann['video_id']]

file.close()
file = open("data/annotations/instances_val_sub.json",'w')
json.dump(data, file)
