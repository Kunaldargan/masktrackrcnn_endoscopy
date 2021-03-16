import json
file = open("instances_train_sub.json",'r')
data = json.load(file)

CATEGORIES = data["categories"]
LabelId = {}
for cat in CATEGORIES:
        LabelId[cat['id']] = cat['name']



for vid in data["videos"]:
	print("###########################################")
	print(len(vid["file_names"]),vid["id"])
	for ann in data["annotations"]:
		if vid["id"] == ann["video_id"]:
			print("segms length:",len(ann["segmentations"]))
			print("bboxes length:",len(ann["bboxes"]))
			print("areas length:",len(ann["areas"]))
			print("annotation id:",ann["id"], LabelId[ann['category_id']])	

