import os
import json

file = open("../output_labels/output_endoVis.pkl.json",'r')
data = json.load(file)

print(data[0].keys())

cat = {}
for d in data:
	if d['category_id'] in cat.keys() and d['score']> cat[d['category_id']]:
		cat[d['category_id']]= d['score']
	elif d['category_id'] not in cat.keys():
		cat[d['category_id']]= d['score']

print(cat)
