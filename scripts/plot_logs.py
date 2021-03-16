import os
import json
import pandas as pd

def get_data(words):
	data = {}
	data["Timestamp"] = words[0]
	data["learning Rate"]=words[1].split(":")[1].strip()
	data["loss_rpn_cls"]=float(words[4].split(":")[1].strip())
	data["loss_rpn_reg"]=float(words[5].split(":")[1].strip())
	data["loss_cls"]=float(words[6].split(":")[1].strip())
	data["acc"]=float(words[7].split(":")[1].strip())
	data["loss_reg"]=float(words[8].split(":")[1].strip())
	data["loss_match"]=float(words[9].split(":")[1].strip())
	data["match_acc"]=float(words[10].split(":")[1].strip())
	data["loss_mask"]=float(words[11].split(":")[1].strip())
	data["loss"]=float(words[12].split(":")[1].strip())

	return data

file = open("work_dirs/masktrack_rcnn_r50_fpn_1x_youtubevos/20201203_220625.log",'r')

df = []
for line in file.readlines():
	words = line.split(",")
	if "Epoch" in line:
		words[1] = words[1].split("\t")[1]
		#print(words)
		df.append(get_data(words))

df = pd.DataFrame(df)
df.to_csv("endoVis.csv")
axes = df.plot.line()#subplots=True)
fig=axes[0].get_figure()
fig.savefig("plots/loss_plot.png")
print(pd.DataFrame(df))
		




