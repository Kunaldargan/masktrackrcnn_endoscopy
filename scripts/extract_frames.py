import cv2
import os
file_path = "./1_Shortlisted_Videos/"
files = os.listdir(file_path)#["Dr_Gaurav_Varshney_0deg_LeftTilt_endo.avi","Dr_Gopal_0Deg_RT_Aux.avi"]

for vid in os.listdir(file_path):
	if vid in files:
		file_name = vid.split(".")[0]
		os.mkdir(os.path.join(file_path,file_name))
		vid_path = os.path.join(file_path, vid)
		cap = cv2.VideoCapture(vid_path)
	
		val_counter=0
		while(cap.isOpened()):
			ret, frame = cap.read()
			val_counter += 1
			if not ret:
            			break
			elif not (val_counter+2)%3:
				#frame = cv2.resize(frame, (1280,1024))
				cv2.imwrite(os.path.join(file_path,file_name)+"/"+str(val_counter)+".jpg",frame)

		cap.release()
