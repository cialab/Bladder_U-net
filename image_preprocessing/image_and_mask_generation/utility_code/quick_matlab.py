import os

path_img = "image/"

for file in os.listdir(path_img):
	file_name, file_extension = os.path.splitext(file)
	if file_extension == ".tiff":
		print(" getMasks(\"image/" + file + "\",\"label/"+file_name+".xml" +"\",\"mask/"+file_name+".png" +"\")")