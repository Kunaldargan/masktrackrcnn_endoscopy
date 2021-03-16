import os

dir = "data/frames"

for fn in os.listdir(dir):
	os.system("ffmpeg -r 2 -i "+"data/frames/"+fn+"/%d.jpg -c:v libx264 data/videos/"+fn+".mp4 -y")		
