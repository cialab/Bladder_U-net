import os

path_img = "val/"

for file in os.listdir(path_img):
	file_name, file_extension = os.path.splitext(file)
	if file_extension != ".db":
		print("tileImages(\"val/"+file+"\",\"val_tiles_512/"+file_name+".png\")")

